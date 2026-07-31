"""
main.py — CLI entry point for the Conversational Note-Taking Agent.

Run this file to start a chat session in your terminal:
  python main.py

Type your message and press Enter. Type 'exit' to quit.
"""

import sys
from agent import Agent
from state import create_initial_state
from store import NoteStore
from vector_store import VectorStore


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
    # Set up the two storage layers
    store = NoteStore(db_path="notes.db")       # SQLite: stores all note data
    vector_store = VectorStore()                 # Qdrant: stores note embeddings for semantic search

    # Create the agent, wiring it to both stores
    agent = Agent(store=store, vector_store=vector_store)

    # Start with a clean blank state (empty memory, IDLE mode)
    state = create_initial_state()

    print_banner()

    while True:
        try:
            # Show the current mode if we're waiting for input (disambiguation or confirmation)
            mode = state.get("mode", "IDLE")
            prefix = f"[{mode}] " if mode != "IDLE" else ""

            user_input = input(f"{prefix}You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                sys.exit(0)

            # Pass the message to the agent and get back the updated state
            state = agent.run(user_input, state)

            # Print the agent's response
            print(f"\nAgent: {state.get('final_response', '')}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()
