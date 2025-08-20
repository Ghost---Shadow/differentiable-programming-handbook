import struct
import math


class MD5:
    def __init__(self):
        # MD5 initial hash values (in little-endian format)
        self.h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476]

        # MD5 constants (sine-based values)
        self.k = [int(abs(math.sin(i + 1)) * (2**32)) & 0xFFFFFFFF for i in range(64)]

        # Rotation amounts for each round
        self.r = (
            [7, 12, 17, 22] * 4
            + [5, 9, 14, 20] * 4
            + [4, 11, 16, 23] * 4
            + [6, 10, 15, 21] * 4
        )

    def _left_rotate(self, value, shift):
        """Left rotate a 32-bit integer by shift bits."""
        return ((value << shift) | (value >> (32 - shift))) & 0xFFFFFFFF

    def _f(self, x, y, z, round_num):
        """MD5 auxiliary functions F, G, H, I."""
        if round_num < 16:
            return (x & y) | (~x & z)
        elif round_num < 32:
            return (x & z) | (y & ~z)
        elif round_num < 48:
            return x ^ y ^ z
        else:
            return y ^ (x | ~z)

    def _get_message_index(self, round_num):
        """Get the message block index for each round."""
        if round_num < 16:
            return round_num
        elif round_num < 32:
            return (5 * round_num + 1) % 16
        elif round_num < 48:
            return (3 * round_num + 5) % 16
        else:
            return (7 * round_num) % 16

    def _pad_message(self, message):
        """Pad the message according to MD5 specification."""
        # Convert string to bytes if necessary
        if isinstance(message, str):
            message = message.encode("utf-8")

        # Original message length in bits
        original_length = len(message) * 8

        # Append the '1' bit (plus seven '0' bits, making it 0x80)
        message += b"\x80"

        # Pad with zeros until message length ≡ 448 (mod 512)
        while len(message) % 64 != 56:
            message += b"\x00"

        # Append the original length as 64-bit little-endian integer
        message += struct.pack("<Q", original_length)

        return message

    def _process_block(self, block):
        """Process a 512-bit block of the message."""
        # Break chunk into sixteen 32-bit little-endian words
        w = list(struct.unpack("<16I", block))

        # Initialize hash value for this chunk
        a, b, c, d = self.h

        # Main MD5 algorithm loop
        for i in range(64):
            f_val = self._f(b, c, d, i)
            g = self._get_message_index(i)

            # Calculate new values
            temp = d
            d = c
            c = b
            b = (
                b
                + self._left_rotate(
                    (a + f_val + self.k[i] + w[g]) & 0xFFFFFFFF, self.r[i]
                )
            ) & 0xFFFFFFFF
            a = temp

        # Add this chunk's hash to result so far
        self.h[0] = (self.h[0] + a) & 0xFFFFFFFF
        self.h[1] = (self.h[1] + b) & 0xFFFFFFFF
        self.h[2] = (self.h[2] + c) & 0xFFFFFFFF
        self.h[3] = (self.h[3] + d) & 0xFFFFFFFF

    def digest(self, message):
        """Calculate MD5 hash of the message."""
        # Reset hash values
        self.h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476]

        # Pad the message
        padded_message = self._pad_message(message)

        # Process each 512-bit block
        for i in range(0, len(padded_message), 64):
            block = padded_message[i : i + 64]
            self._process_block(block)

        # Produce the final hash value as a 128-bit number (little-endian)
        return struct.pack("<4I", *self.h)

    def hexdigest(self, message):
        """Calculate MD5 hash and return as hexadecimal string."""
        return self.digest(message).hex()


def md5_hash(message):
    """Convenience function to calculate MD5 hash."""
    md5 = MD5()
    return md5.hexdigest(message)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "",
        "a",
        "abc",
        "message digest",
        "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "The quick brown fox jumps over the lazy dog",
    ]

    # Expected MD5 hashes for verification
    expected = [
        "d41d8cd98f00b204e9800998ecf8427e",
        "0cc175b9c0f1b6a831c399e269772661",
        "900150983cd24fb0d6963f7d28e17f72",
        "f96b697d7cb7938d525a2f31aaf161d0",
        "c3fcd3d76192e4007dfb496cca67e13b",
        "d174ab98d277d9f5a5611c2c9f419d9f",
        "9e107d9d372bb6826bd81d3542a419d6",
    ]

    print("MD5 Hash Implementation Test Results:")
    print("=" * 50)

    for i, test_message in enumerate(test_cases):
        result = md5_hash(test_message)
        status = "✓ PASS" if result == expected[i] else "✗ FAIL"

        print(f"Input: '{test_message}'")
        print(f"Output: {result}")
        print(f"Expected: {expected[i]}")
        print(f"Status: {status}")
        print("-" * 30)

    # Interactive test
    print("\nYou can also test with custom input:")
    custom_input = input("Enter a message to hash (or press Enter to skip): ")
    if custom_input:
        custom_hash = md5_hash(custom_input)
        print(f"MD5 hash of '{custom_input}': {custom_hash}")
