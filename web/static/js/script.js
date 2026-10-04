const sendBtn = document.getElementById('send-btn');
const userInput = document.getElementById('user-input');
const chatWindow = document.getElementById('chat-window');

       
marked.setOptions({
    breaks: true,
    gfm: true
});

async function handleSend() {
    const message = userInput.value.trim();
    if (!message) return;

   
    const userBubble = document.createElement('div');
    userBubble.className = 'user-bubble';
    userBubble.textContent = '👤 ' + message;
    chatWindow.appendChild(userBubble);
    userInput.value = '';
    chatWindow.scrollTop = chatWindow.scrollHeight;

   
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        // 3. ANALYSE DU MARKDOWN 
        
        const formattedResponse = marked.parse(data.response);

        chatWindow.innerHTML += `<div class="bot-bubble">🧠 ${formattedResponse}</div>`;
        chatWindow.scrollTop = chatWindow.scrollHeight;
    } catch (error) {
        console.error("Erreur de connexion:", error);
        chatWindow.innerHTML += `<div class="bot-bubble" style="border-left-color: red;">❌ Erreur : Impossible de contacter le serveur.</div>`;
    }
}

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});
