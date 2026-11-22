
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import json
from pathlib import Path

from src.chatbot import ChatBot


class ChatbotGUI:

    # Midnight Purple Theme
    BG = "#130D20"
    HEADER = "#211432"
    CHAT_BG = "#1D142B"
    INPUT_BG = "#2B1D3D"
    PURPLE = "#9333EA"
    PURPLE_HOVER = "#A855F7"
    LIGHT_PURPLE = "#D8B4FE"
    USER_COLOR = "#C084FC"
    BOT_COLOR = "#4ADE80"
    TEXT_COLOR = "#F5F3FF"
    MUTED = "#A78BBA"
    RED = "#F87171"

    def __init__(self, root):

        self.root = root
        self.root.title("Intent-Based AI Chatbot")
        self.root.geometry("950x700")
        self.root.minsize(700, 550)
        self.root.configure(bg=self.BG)

        self.chatbot = ChatBot()

        self.placeholder = "Type your message here..."

        # Persistent chat history location
        self.history_file = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "chat_history.json"
        )

        self.history_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.chat_history = []

        self.setup_styles()
        self.create_header()
        self.create_chat_area()
        self.create_input_area()

        # Restore previous messages
        self.load_chat_history()

        if not self.chat_history:

            self.display_message(
                "Bot",
                "Hello! Welcome to Intent-Based AI Chatbot. How can I help you?"
            )

    # --------------------------------------------------
    # STYLES
    # --------------------------------------------------

    def setup_styles(self):

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Purple.Vertical.TScrollbar",
            background="#49305F",
            troughcolor=self.BG,
            bordercolor=self.BG,
            arrowcolor=self.LIGHT_PURPLE,
            width=12
        )

    # --------------------------------------------------
    # BUTTON HOVER EFFECT
    # --------------------------------------------------

    def add_hover_effect(
        self,
        button,
        normal_color,
        hover_color
    ):

        button.bind(
            "<Enter>",
            lambda event: button.config(
                bg=hover_color
            )
        )

        button.bind(
            "<Leave>",
            lambda event: button.config(
                bg=normal_color
            )
        )

    # --------------------------------------------------
    # HEADER
    # --------------------------------------------------

    def create_header(self):

        header = tk.Frame(
            self.root,
            bg=self.HEADER,
            height=115,
            highlightbackground="#53346F",
            highlightthickness=1
        )

        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title_frame = tk.Frame(
            header,
            bg=self.HEADER
        )

        title_frame.pack(pady=(17, 3))

        robot_icon = tk.Label(
            title_frame,
            text="🤖",
            font=("Segoe UI Emoji", 25),
            bg=self.HEADER,
            fg="white"
        )

        robot_icon.pack(side=tk.LEFT, padx=8)

        title = tk.Label(
            title_frame,
            text="AI CHATBOT",
            font=("Arial", 23, "bold"),
            fg=self.LIGHT_PURPLE,
            bg=self.HEADER
        )

        title.pack(side=tk.LEFT)

        subtitle = tk.Label(
            header,
            text="✦  Intent-Based Chatbot using NLP and Deep Learning  ✦",
            font=("Arial", 10),
            fg=self.MUTED,
            bg=self.HEADER
        )

        subtitle.pack()

        status = tk.Label(
            header,
            text="●  ONLINE",
            font=("Arial", 9, "bold"),
            fg=self.BOT_COLOR,
            bg=self.HEADER
        )

        status.pack(pady=(5, 0))

    # --------------------------------------------------
    # CHAT AREA
    # --------------------------------------------------

    def create_chat_area(self):

        container = tk.Frame(
            self.root,
            bg=self.BG,
            padx=20,
            pady=15
        )

        container.pack(
            fill=tk.BOTH,
            expand=True
        )

        chat_border = tk.Frame(
            container,
            bg="#49305F",
            padx=2,
            pady=2
        )

        chat_border.pack(
            fill=tk.BOTH,
            expand=True
        )

        self.chat_area = tk.Text(
            chat_border,
            bg=self.CHAT_BG,
            fg=self.TEXT_COLOR,
            font=("Arial", 12),
            wrap=tk.WORD,
            padx=18,
            pady=18,
            bd=0,
            state=tk.DISABLED,
            cursor="arrow",
            insertbackground="white",
            selectbackground=self.PURPLE
        )

        scrollbar = ttk.Scrollbar(
            chat_border,
            orient=tk.VERTICAL,
            command=self.chat_area.yview,
            style="Purple.Vertical.TScrollbar"
        )

        self.chat_area.configure(
            yscrollcommand=scrollbar.set
        )

        scrollbar.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )

        self.chat_area.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        # Message formatting
        self.chat_area.tag_configure(
            "user",
            foreground=self.USER_COLOR,
            font=("Arial", 11, "bold")
        )

        self.chat_area.tag_configure(
            "bot",
            foreground=self.BOT_COLOR,
            font=("Arial", 11, "bold")
        )

        self.chat_area.tag_configure(
            "message",
            foreground=self.TEXT_COLOR,
            font=("Arial", 11),
            spacing1=5,
            spacing3=5
        )

        self.chat_area.tag_configure(
            "time",
            foreground=self.MUTED,
            font=("Arial", 9)
        )

        self.chat_area.tag_configure(
            "details",
            foreground=self.LIGHT_PURPLE,
            font=("Arial", 9, "italic")
        )

    # --------------------------------------------------
    # INPUT AREA
    # --------------------------------------------------

    def create_input_area(self):

        input_frame = tk.Frame(
            self.root,
            bg=self.HEADER,
            padx=20,
            pady=18,
            highlightbackground="#53346F",
            highlightthickness=1
        )

        input_frame.pack(fill=tk.X)

        self.message_entry = tk.Entry(
            input_frame,
            font=("Arial", 12),
            bg=self.INPUT_BG,
            fg=self.MUTED,
            insertbackground="white",
            relief=tk.SUNKEN,
            bd=2,
            highlightthickness=1,
            highlightbackground="#604477",
            highlightcolor=self.PURPLE
        )

        self.message_entry.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            ipady=12,
            padx=(0, 12)
        )

        self.message_entry.insert(
            0,
            self.placeholder
        )

        self.message_entry.bind(
            "<FocusIn>",
            self.clear_placeholder
        )

        self.message_entry.bind(
            "<FocusOut>",
            self.restore_placeholder
        )

        self.message_entry.bind(
            "<Return>",
            self.send_message
        )

        # Send button
        send_button = tk.Button(
            input_frame,
            text="✨  SEND",
            font=("Arial", 10, "bold"),
            bg=self.PURPLE,
            fg="white",
            activebackground=self.PURPLE_HOVER,
            activeforeground="white",
            relief=tk.RAISED,
            bd=4,
            padx=18,
            pady=7,
            cursor="hand2",
            command=self.send_message
        )

        send_button.pack(side=tk.RIGHT)

        self.add_hover_effect(
            send_button,
            self.PURPLE,
            self.PURPLE_HOVER
        )

        # Bottom buttons
        button_frame = tk.Frame(
            self.root,
            bg=self.BG,
            pady=12
        )

        button_frame.pack(fill=tk.X)

        # Clear Chat button
        clear_button = tk.Button(
            button_frame,
            text="🧹  Clear Chat",
            font=("Arial", 10, "bold"),
            bg="#49305F",
            fg="#E9D5FF",
            activebackground="#63447F",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            padx=15,
            pady=7,
            cursor="hand2",
            command=self.clear_chat
        )

        clear_button.pack(
            side=tk.LEFT,
            padx=20
        )

        self.add_hover_effect(
            clear_button,
            "#49305F",
            "#63447F"
        )

        # Exit button
        exit_button = tk.Button(
            button_frame,
            text="🚪  EXIT",
            font=("Arial", 10, "bold"),
            bg="#55243D",
            fg="#FCA5A5",
            activebackground="#7F1D3D",
            activeforeground="white",
            relief=tk.RAISED,
            bd=3,
            padx=15,
            pady=7,
            cursor="hand2",
            command=self.root.destroy
        )

        exit_button.pack(
            side=tk.RIGHT,
            padx=20
        )

        self.add_hover_effect(
            exit_button,
            "#55243D",
            "#7F1D3D"
        )

        self.message_entry.focus_set()

    # --------------------------------------------------
    # PLACEHOLDER
    # --------------------------------------------------

    def clear_placeholder(self, event=None):

        if self.message_entry.get() == self.placeholder:

            self.message_entry.delete(0, tk.END)

            self.message_entry.config(
                fg=self.TEXT_COLOR
            )

    def restore_placeholder(self, event=None):

        if not self.message_entry.get().strip():

            self.message_entry.delete(0, tk.END)

            self.message_entry.insert(
                0,
                self.placeholder
            )

            self.message_entry.config(
                fg=self.MUTED
            )

    # --------------------------------------------------
    # SAVE CHAT HISTORY
    # --------------------------------------------------

    def save_chat_history(self):

        try:

            with open(
                self.history_file,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    self.chat_history,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

        except (OSError, TypeError) as error:

            print("History Save Error:", error)

    # --------------------------------------------------
    # LOAD CHAT HISTORY
    # --------------------------------------------------

    def load_chat_history(self):

        if not self.history_file.exists():

            self.chat_history = []
            return

        try:

            with open(
                self.history_file,
                "r",
                encoding="utf-8-sig"
            ) as file:

                data = json.load(file)

            if not isinstance(data, list):

                raise ValueError(
                    "Invalid chat history format"
                )

            self.chat_history = [
                item for item in data
                if isinstance(item, dict)
                and item.get("sender") in ("You", "Bot")
                and isinstance(item.get("message"), str)
                and isinstance(item.get("timestamp"), str)
            ]

            # Restore messages
            for item in self.chat_history:

                self.display_message(
                    item["sender"],
                    item["message"],
                    item.get("details"),
                    timestamp=item["timestamp"],
                    save=False
                )

        except (
            OSError,
            json.JSONDecodeError,
            ValueError
        ) as error:

            print("History Load Error:", error)

            self.chat_history = []

    # --------------------------------------------------
    # DISPLAY MESSAGE
    # --------------------------------------------------

    def display_message(
        self,
        sender,
        message,
        details=None,
        timestamp=None,
        save=True
    ):

        if timestamp is None:

            timestamp = datetime.now().strftime(
                "%I:%M %p"
            )

        self.chat_area.configure(
            state=tk.NORMAL
        )

        tag = "user" if sender == "You" else "bot"

        icon = "👤" if sender == "You" else "🤖"

        self.chat_area.insert(
            tk.END,
            f"\n{icon}  {sender}  ",
            tag
        )

        self.chat_area.insert(
            tk.END,
            f"{timestamp}\n",
            "time"
        )

        self.chat_area.insert(
            tk.END,
            f"{message}\n",
            "message"
        )

        if details:

            self.chat_area.insert(
                tk.END,
                f"{details}\n",
                "details"
            )

        self.chat_area.insert(
            tk.END,
            "\n"
        )

        self.chat_area.configure(
            state=tk.DISABLED
        )

        self.chat_area.see(
            tk.END
        )

        # Store message in JSON
        if save:

            self.chat_history.append({

                "sender": sender,
                "message": message,
                "details": details,
                "timestamp": timestamp

            })

            self.save_chat_history()

    # --------------------------------------------------
    # SEND MESSAGE
    # --------------------------------------------------

    def send_message(self, event=None):

        user_message = self.message_entry.get().strip()

        if (
            not user_message
            or user_message == self.placeholder
        ):

            return "break"

        self.display_message(
            "You",
            user_message
        )

        self.message_entry.delete(
            0,
            tk.END
        )

        self.message_entry.config(
            fg=self.TEXT_COLOR
        )

        try:

            # Get response from existing chatbot
            result = self.chatbot.get_response(
                user_message
            )

            bot_response = result["response"]

            intent = result["intent"]

            confidence = result["confidence"]

            details = (
                f"✦ Intent: {intent}  |  "
                f"Confidence: {confidence * 100:.2f}%"
            )

        except Exception as error:

            bot_response = (
                "Sorry, something went wrong. Please try again."
            )

            details = None

            print(
                "Chatbot Error:",
                error
            )

        self.display_message(
            "Bot",
            bot_response,
            details
        )

        self.message_entry.focus_set()

        return "break"

    # --------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------

    def clear_chat(self):

        self.chat_area.configure(
            state=tk.NORMAL
        )

        self.chat_area.delete(
            "1.0",
            tk.END
        )

        self.chat_area.configure(
            state=tk.DISABLED
        )

        # Clear saved conversation
        self.chat_history = []

        self.save_chat_history()

        self.display_message(
            "Bot",
            "Chat cleared! How can I help you?"
        )

        self.message_entry.delete(
            0,
            tk.END
        )

        self.message_entry.config(
            fg=self.TEXT_COLOR
        )

        self.message_entry.focus_set()


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    root = tk.Tk()

    app = ChatbotGUI(root)

    root.mainloop()
