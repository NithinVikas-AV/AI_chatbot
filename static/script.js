const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");
const loading = document.getElementById("loading");
const csvInput = document.getElementById("csvFile");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const userMessage = input.value.trim();
    if (!userMessage || userMessage.length > 500) {
        addMessage("bot", "Please enter a message (max 500 characters).");
        return;
    }
    addMessage("user", userMessage);
    input.value = "";
    loading.style.display = "block";

    try {
        const response = await fetch("http://127.0.0.1:5000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage })
        });
        const data = await response.json();
        addMessage("bot", data.reply);
    } catch (error) {
        addMessage("bot", "Failed to get response. Please try again.");
    } finally {
        loading.style.display = "none";
    }
});

csvInput.addEventListener("change", async () => {
    const file = csvInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:5000/upload", {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            addMessage("bot", `✅ File "${file.name}" uploaded successfully.`);
        } else {
            addMessage("bot", "❌ File upload failed.");
        }
    } catch (error) {
        addMessage("bot", "⚠️ Upload error. Please try again.");
    }
});

function addMessage(sender, text) {
    const msg = document.createElement("div");
    msg.className = `message ${sender}`;
    msg.innerText = text;
    chatBox.appendChild(msg);
    msg.scrollIntoView({ behavior: "smooth" });
}
