"""
===============================================================================================
# Groq API Chatbot Example
This script demonstrates how to use the Groq API to create a simple chat application.

It initializes a conversation with a system message and allows the user to ask questions.
The bot responds based on the conversation history and the Groq API's response.
===============================================================================================
"""


from groq import Groq

client = Groq(api_key="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")

# Initialize conversation history
conversation_history = [{"role": "system", "content": "Act like a helpful assistant. Give a simple and clear answer. If you don't know the answer, say 'I don't know'."}]

def call_groq_api(user_query):
    try:
        # Add user query to conversation historywha
        conversation_history.append({"role": "user", "content": user_query})
        
        # Call the API with the updated conversation history
        chat_completion = client.chat.completions.create(
            messages=conversation_history,
            model="llama3-8b-8192",
            temperature=0.1  # Lower temperature for deterministic output
        )
        
        # Get the bot's response
        answer = chat_completion.choices[0].message.content.strip()
        
        # Add bot's response to conversation history
        conversation_history.append({"role": "assistant", "content": answer})
        
        print("Bot:", answer)
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return "Sorry, I couldn't process your request at the moment."
    
def get_user_input():
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        return
    call_groq_api(user_input)
    get_user_input()

if __name__ == "__main__":
    get_user_input()