const API_BASE = "http://localhost:8000";

document.getElementById("send").addEventListener("click", async () => {
  const input = document.getElementById("input").value;
  const output = document.getElementById("output");

  output.textContent = "Thinking...";

  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: input,
        execute: true
      })
    });

    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    output.textContent = "Error: " + e.message;
  }
});

// ---------- Voice (mic) ----------
let mediaRecorder;
let audioChunks = [];

document.getElementById("mic").addEventListener("click", async () => {
  const output = document.getElementById("output");

  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      const formData = new FormData();
      formData.append("file", blob, "recording.webm");

      output.textContent = "Uploading audio...";
      const res = await fetch(`${API_BASE}/generate-audio`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      output.textContent = JSON.stringify(data, null, 2);
    };

    mediaRecorder.start();
    output.textContent = "Recording... click mic again to stop.";
  } catch (err) {
    output.textContent = "Microphone error: " + err.message;
  }
});

// ---------- Camera ----------
let cameraStream;

document.getElementById("camera").addEventListener("click", async () => {
  const video = document.getElementById("cameraPreview");
  const captureBtn = document.getElementById("capture");
  const output = document.getElementById("output");

  if (video.style.display === "block") {
    // Stop camera
    video.style.display = "none";
    captureBtn.style.display = "none";
    cameraStream?.getTracks().forEach((t) => t.stop());
    cameraStream = null;
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = cameraStream;
    video.style.display = "block";
    captureBtn.style.display = "inline-block";
    output.textContent = "Camera active. Click capture to send image.";
  } catch (err) {
    output.textContent = "Camera error: " + err.message;
  }
});

document.getElementById("capture").addEventListener("click", async () => {
  const output = document.getElementById("output");
  const video = document.getElementById("cameraPreview");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "capture.png");

    output.textContent = "Uploading image...";
    const res = await fetch(`${API_BASE}/generate-image`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  }, "image/png");
});

document.getElementById("send").addEventListener("click", async () => {
  const input = document.getElementById("input").value;
  const output = document.getElementById("output");

  output.textContent = "Thinking...";

  try {
    const res = await fetch("http://localhost:8000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
         message: input,
         execute: true
        })
    });

    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    output.textContent = "Error: " + e.message;
  }
});
