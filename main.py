"""
CLI Interactive Interface for Conversational Note-Taking Agent.

Run this script to start an interactive terminal session with the agent.
"""

import sys
from dotenv import load_dotenv
from graph import NoteAgentGraph
from state import create_initial_state
from store import NoteStore
from vector_store import VectorNoteStore


def print_banner():
    print("=" * 65)
    print("      Conversational Note-Taking Agent — TechLabs London CLI")
    print("=" * 65)
    print("  • Save notes  : 'Save a note about team standup on Tuesdays'")
    print("  • Search notes: 'What did I write about standup?'")
    print("  • Update notes: 'Update my standup note to Wednesdays'")
    print("  • Delete notes: 'Delete the note about old office'")
    print("  • Type 'exit' or 'quit' to terminate session.")
    print("=" * 65 + "\n")


def main():
    # Load environment variables (.env)
    load_dotenv()

    # Initialize storage & graph services
    store = NoteStore(db_path="notes.db")
    vstore = VectorNoteStore(vector_dim=128)
    graph = NoteAgentGraph(store=store, vector_store=vstore)


    state = create_initial_state()

    print_banner()

    while True:
        try:
            mode_indicator = f"[{state.get('mode', 'IDLE')}] " if state.get("mode") != "IDLE" else ""
            user_input = input(f"{mode_indicator}User: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                sys.exit(0)

            # Step the LangGraph state machine
            state = graph.run(user_input, state)

            response = state.get("final_response", "")
            print(f"\nAgent: {response}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()
