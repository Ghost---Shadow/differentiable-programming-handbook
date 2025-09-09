import torch
import torch.nn.functional as F

# Define the number of possible turns in a tic-tac-toe game
num_turns = 5

# Create raw logits tensor: [num_turns, 3, 3, 3]
# - First dimension: different turns/time steps
# - Second and third dimensions: 3x3 board
# - Fourth dimension: logits for [X, O, empty]
game_logits = torch.randn(num_turns, 3, 3, 3, requires_grad=True)

# Initialize logits to favor empty positions
with torch.no_grad():
    game_logits[:, :, :, 2] = 2.0  # Higher logit for empty

# Turn order constraint: X plays on even turns, O on odd turns
turn_order_mask = torch.zeros(num_turns, 3)  # Shape: [turns, encoding]
turn_order_mask[::2, 0] = 1.0  # X can only play on even turns (0,2,4,6,8)
turn_order_mask[1::2, 1] = 1.0  # O can only play on odd turns (1,3,5,7)
turn_order_mask[:, 2] = 1.0  # Empty is always allowed

# Target game state - example: X wins with top row
# Each cell shows [X_count, O_count, Empty_count] across all turns
target_state = torch.zeros(3, 3, 3)
target_state[:, :, 2] = 1.0  # (1,2) empty all turns
# Top row: X occupies positions (0,0), (0,1), (0,2) once each
target_state[0, 0, 0] = 1.0  # X at (0,0)
target_state[0, 1, 0] = 1.0  # X at (0,1)
target_state[0, 2, 0] = 1.0  # X at (0,2)
# O makes some moves
target_state[1, 0, 1] = 1.0  # O at (1,0)
target_state[1, 1, 1] = 1.0  # O at (1,1)
# Remaining cells stay empty across all turns


def softmax_deviation_loss(raw_values, dim=-1):
    """Measure how far raw values deviate from their softmax normalization"""
    return F.mse_loss(F.softmax(raw_values, dim=dim), raw_values)


def forward_pass(game_logits, turn_order_mask):
    """Compute forward pass and all constraint losses"""
    # Apply turn order mask and softmax (broadcast mask across spatial dims)
    masked_logits = game_logits * turn_order_mask.unsqueeze(1).unsqueeze(2)
    game_tensor = F.softmax(masked_logits, dim=-1)

    # Probability normalization constraint: how far game_tensor deviates from softmax
    probability_normalization_loss = softmax_deviation_loss(game_tensor, dim=-1)

    # Single move per turn + turn order constraint
    x_totals = torch.sum(game_tensor[:, :, :, 0], dim=(1, 2))
    o_totals = torch.sum(game_tensor[:, :, :, 1], dim=(1, 2))
    move_totals = torch.stack([x_totals, o_totals], dim=-1)  # Shape: [turns, 2]

    # Use existing turn_order_mask (only X and O columns, skip empty)
    expected_moves = turn_order_mask[:, :2]  # Shape: [turns, 2]

    single_move_per_turn_loss = F.mse_loss(move_totals, expected_moves)

    # Cell exclusivity constraint: each cell can only have X or O, never both
    x_cell_totals = torch.sum(game_tensor[:, :, :, 0], dim=0)
    o_cell_totals = torch.sum(game_tensor[:, :, :, 1], dim=0)
    cell_totals = torch.stack(
        [x_cell_totals, o_cell_totals], dim=-1
    )  # Shape: [3, 3, 2]
    cell_exclusivity_loss = softmax_deviation_loss(cell_totals, dim=-1)

    # Final game state: get the final board state (who occupies each cell)
    x_moves = game_tensor[:, :, :, 0]  # [turns, 3, 3]
    o_moves = game_tensor[:, :, :, 1]  # [turns, 3, 3]

    # Find which turn each cell was played (if any)
    x_played = torch.sum(x_moves, dim=0)  # Total X probability per cell
    o_played = torch.sum(o_moves, dim=0)  # Total O probability per cell

    # Final state: [X_occupied, O_occupied, Empty]
    final_game_state = torch.stack(
        [x_played, o_played, torch.ones_like(x_played) - x_played - o_played], dim=-1
    )

    return (
        game_tensor,
        final_game_state,
        probability_normalization_loss,
        single_move_per_turn_loss,
        cell_exclusivity_loss,
    )


# Training with SGD
# optimizer = torch.optim.SGD([game_logits], lr=0.1)
optimizer = torch.optim.AdamW([game_logits])
mse_loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(10000):
    optimizer.zero_grad()

    # Forward pass
    (
        game_tensor,
        final_game_state,
        prob_norm_loss,
        single_move_loss,
        cell_excl_loss,
    ) = forward_pass(game_logits, turn_order_mask)

    # Total loss: target matching + constraint penalties
    target_matching_loss = mse_loss_fn(final_game_state, target_state)
    constraint_penalties = prob_norm_loss + single_move_loss + cell_excl_loss
    total_loss = target_matching_loss + constraint_penalties

    # Backward pass
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        game_logits.data = F.softmax(game_logits.data, dim=-1)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:4d} | Total: {total_loss.item():.4f} | "
            f"Target: {target_matching_loss.item():.4f} | "
            f"prob_norm_loss: {prob_norm_loss.item():.4f} | "
            f"single_move_loss: {single_move_loss.item():.4f} | "
            f"cell_excl_loss: {cell_excl_loss.item():.4f} | "
        )

print(f"\nFinal total loss: {total_loss.item():.4f}")
print(f"Final target loss: {target_matching_loss.item():.4f}")
print(f"Final constraint loss: {constraint_penalties.item():.4f}")


print(game_logits)
print(target_state.argmax(-1))
print(final_game_state.argmax(-1))
