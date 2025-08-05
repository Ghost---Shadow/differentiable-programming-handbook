import winsound


def play_notification():
    """Play a system notification sound"""
    # Option 1: Play default system sound
    winsound.MessageBeep(winsound.MB_OK)

    # Option 2: Play Windows default beep
    # winsound.Beep(1000, 500)  # frequency: 1000Hz, duration: 500ms


def play_custom_sound():
    """Play a custom WAV file (if you have one)"""
    try:
        # Replace 'notification.wav' with path to your sound file
        winsound.PlaySound("notification.wav", winsound.SND_FILENAME)
    except:
        print("Could not play custom sound file")
        # Fallback to system sound
        winsound.MessageBeep(winsound.MB_OK)


if __name__ == "__main__":
    print("Playing notification sound...")
    play_notification()
    print("Sound played!")
