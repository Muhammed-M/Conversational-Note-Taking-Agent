"""
main.py — CLI entry point for the Conversational Note-Taking Agent.

Run this file from the project root:
  python main.py
"""

import sys
from src.agent import Agent
from src.state import create_initial_state
from src.store import NoteStore
from src.vector_store import VectorStore


def print_banner():
    print("=" * 65)
    print("      Conversational Note-Taking Agent")
    print("=" * 65)
    print("  Save  : 'Save a note about our standup meeting every Tuesday'")
    print("  Search: 'What did I write about the standup?'")
    print("  Update: 'Update my standup note to say Wednesdays'")
    print("  Delete: 'Delete the standup note'")
    print("  Quit  : type 'exit' or 'quit'")
    print("=" * 65 + "\n")


def main():
    store = NoteStore()
    vector_store = VectorStore()
    agent = Agent(store=store, vector_store=vector_store)
    state = create_initial_state()

    print_banner()

    while True:
        try:
            mode = state.get("mode", "IDLE")
            prefix = f"[{mode}] " if mode != "IDLE" else ""

            user_input = input(f"{prefix}You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                sys.exit(0)

            state = agent.run(user_input, state)
            print(f"\nAgent: {state.get('final_response', '')}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
