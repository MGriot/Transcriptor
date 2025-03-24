import json

chunk_files = [
    "out\\chunk_0.json",
    "out\\chunk_1.json",
    "out\\chunk_2.json",
    "out\\chunk_3.json",
    "out\\chunk_4.json",
    "out\\chunk_5.json",
]

combined_data = []
for chunk_file in chunk_files:
    try:
        with open(chunk_file, "r") as f:
            chunk_data = json.load(f)
            for segment in chunk_data:
                speaker = segment.get("speaker")
                if speaker is None:
                    print(
                        f"Warning: No speaker found in segment from file {chunk_file}"
                    )
                combined_data.append(
                    {
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "text": segment.get("text"),
                        "speaker": speaker,
                        "words": segment.get("words"),
                    }
                )
    except FileNotFoundError:
        print(f"Warning: Could not find file {chunk_file}")
    except json.JSONDecodeError:
        print(f"Warning: Could not parse JSON in {chunk_file}")

# Print all combined text in sequence with speaker information
print("\n=== Combined Transcription ===\n")
for segment in combined_data:
    speaker = segment.get("speaker", "Unknown Speaker")
    print(f"[{speaker}]: {segment['text']}")
