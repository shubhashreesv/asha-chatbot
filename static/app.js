class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;

        if (openButton) {
            openButton.addEventListener('click', () => this.toggleState(chatBox));
        }

        if (sendButton) {
            sendButton.addEventListener('click', () => this.onSendButton(chatBox));
        }

        const node = chatBox ? chatBox.querySelector('input') : null;
        if (node) {
            node.addEventListener("keyup", ({ key }) => {
                if (key === "Enter") {
                    this.onSendButton(chatBox);
                }
            });
        }
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the box
        if (this.state) {
            chatbox.classList.add('chatbox--active');
        } else {
            chatbox.classList.remove('chatbox--active');
        }
    }

    onSendButton(chatbox) {
        const textField = chatbox ? chatbox.querySelector('input') : null;
        if (!textField) return;

        const text1 = textField.value;
        if (text1 === "") {
            return;
        }

        const msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(r => {
            const msg2 = { name: "Asha", message: r.answer };
            this.messages.push(msg2);
            this.updateChatText(chatbox);
            textField.value = '';
        })
        .catch((error) => {
            console.error('Error:', error);
            this.messages.push({ name: "Asha", message: "There was an error processing your request." });
            this.updateChatText(chatbox);
            textField.value = '';
        });
    }

    updateChatText(chatbox) {
        let html = '';
        this.messages.slice().reverse().forEach(function(item) {
            if (item.name === "Asha") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox ? chatbox.querySelector('.chatbox__messages') : null;
        if (chatmessage) {
            chatmessage.innerHTML = html;
        }
    }
}

const chatbox = new Chatbox();
chatbox.display();
