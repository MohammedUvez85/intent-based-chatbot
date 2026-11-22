
import tkinter as tk
from tkinter import ttk
from datetime import datetime

from chatbot import get_response


class ChatbotGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Intent-Based AI Chatbot")
        self.root.geometry("900x650")
        self.root.minsize(700, 500)
        self.root.configure(bg="#101827")

        self.setup_styles()
        self.create_header()
        self.create_chat_area()
        self.create_input_area()

        self.display_message(
            "Bot",
            "Hello! Welcome to Intent-Based AI Chatbot. How can I help you?"
        )

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Send.TButton",
            font=("Arial", 11, "bold"),
            background="#2563EB",
            foreground="white",
            padding=10
        )

        style.map(
            "Send.TButton",
            background=[("active", "#1D4ED8")]
        )

    def create_header(self):

        header = tk.Frame(
            self.root,
            bg="#172338",
            height=85
        )
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="AI CHATBOT",
            font=("Arial", 22, "bold"),
            fg="white",
            bg="#172338"
        )
        title.pack(pady=(15, 2))

        subtitle = tk.Label(
            header,
            text="Intent-Based Chatbot using NLP and Deep Learning",
            font=("Arial", 10),
            fg="#94A3B8",
            bg="#172338"
        )
        subtitle.pack()

    def create_chat_area(self):

        container = tk.Frame(
            self.root,
            bg="#101827"
        )
        container.pack(
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=15
        )

        self.chat_area = tk.Text(
            container,
            bg="#1E293B",
            fg="white",
            font=("Arial", 12),
            wrap=tk.WORD,
            padx=15,
            pady=15,
            bd=0,
            state=tk.DISABLED,
            cursor="arrow"
        )

        scrollbar = ttk.Scrollbar(
            container,
            orient=tk.VERTICAL,
            command=self.chat_area.yview
        )

        self.chat_area.configure(
            yscrollcommand=scrollbar.set
        )

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_area.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        self.chat_area.tag_configure(
            "user",
            foreground="#60A5FA",
            font=("Arial", 11, "bold")
        )

        self.chat_area.tag_configure(
            "bot",
            foreground="#34D399",
            font=("Arial", 11, "bold")
        )

        self.chat_area.tag_configure(
            "message",
            foreground="#F8FAFC",
            font=("Arial", 11)
        )

        self.chat_area.tag_configure(
            "time",
            foreground="#94A3B8",
            font=("Arial", 9)
        )

    def create_input_area(self):

        input_frame = tk.Frame(
            self.root,
            bg="#172338",
            padx=20,
            pady=15
        )
        input_frame.pack(fill=tk.X)

        self.message_entry = tk.Entry(
            input_frame,
            font=("Arial", 12),
            bg="#334155",
            fg="white",
            insertbackground="white",
            relief=tk.FLAT
        )

        self.message_entry.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            ipady=12,
            padx=(0, 10)
        )

        self.message_entry.insert(
            0,
            "Type your message here..."
        )

        self.message_entry.bind(
            "<FocusIn>",
            self.clear_placeholder
        )

        self.message_entry.bind(
            "<Return>",
            self.send_message
        )

        send_button = ttk.Button(
            input_frame,
            text="Send",
            style="Send.TButton",
            command=self.send_message
        )
        send_button.pack(side=tk.RIGHT)

        clear_button = tk.Button(
            self.root,
            text="Clear Chat",
            font=("Arial", 10),
            bg="#172338",
            fg="#F87171",
            activebackground="#334155",
            activeforeground="white",
            relief=tk.FLAT,
            command=self.clear_chat
        )
        clear_button.pack(pady=(0, 10))

    def clear_placeholder(self, event):

        if self.message_entry.get() == "Type your message here...":
            self.message_entry.delete(0, tk.END)

    def display_message(self, sender, message):

        self.chat_area.configure(state=tk.NORMAL)

        current_time = datetime.now().strftime("%I:%M %p")

        if sender == "You":
            tag = "user"
        else:
            tag = "bot"

        self.chat_area.insert(
            tk.END,
            f"\n{sender}  ",
            tag
        )

        self.chat_area.insert(
            tk.END,
            f"{current_time}\n",
            "time"
        )

        self.chat_area.insert(
            tk.END,
            f"{message}\n",
            "message"
        )

        self.chat_area.configure(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def send_message(self, event=None):

        user_message = self.message_entry.get().strip()

        if not user_message or user_message == "Type your message here...":
            return

        self.display_message("You", user_message)

        self.message_entry.delete(0, tk.END)

        try:
            bot_response = get_response(user_message)
        except Exception as error:
            bot_response = "Sorry, something went wrong. Please try again."
            print("Chatbot Error:", error)

        self.display_message("Bot", bot_response)

    def clear_chat(self):

        self.chat_area.configure(state=tk.NORMAL)
        self.chat_area.delete("1.0", tk.END)
        self.chat_area.configure(state=tk.DISABLED)

        self.display_message(
            "Bot",
            "Chat cleared! How can I help you?"
        )


if __name__ == "__main__":

    root = tk.Tk()

    app = ChatbotGUI(root)

    root.mainloop()
