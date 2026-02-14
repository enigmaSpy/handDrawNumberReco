const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.strokeStyle = "white";
ctx.lineWidth = 20;
ctx.lineCap = "round";

let drawing = false;

canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
  if (!drawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
}

function clearCanvas() {
  ctx.beginPath();
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").innerText = "?";
}

async function sendImage() {
  const blob = await new Promise(resolve =>
    canvas.toBlob(resolve, "image/png")
  );

  const formData = new FormData();
  formData.append("file", blob);

  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    body: formData
  });

  const data = await res.json();
  document.getElementById("result").innerText = data.digit;
}