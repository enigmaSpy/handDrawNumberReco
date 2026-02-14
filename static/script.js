const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#000';  
ctx.lineWidth = 20; 
ctx.lineJoin = 'round';


function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    [lastX, lastY] = getCoordinates(e, rect);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const [x, y] = getCoordinates(e, rect);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
}

function getCoordinates(e, rect) {
    const x = (e.clientX || e.touches[0].clientX) - rect.left;
    const y = (e.clientY || e.touches[0].clientY) - rect.top;
    return [x, y];
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startDrawing(e);
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e);
});
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
});

predictBtn.addEventListener('click', async () => {
    try {
        const blob = await new Promise(resolve => canvas.toBlob(resolve));
        
        const formData = new FormData();
        formData.append('file', blob, 'drawing.png');
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        showResult(data);
        
    } catch (error) {
        showError('Connection error: ' + error.message);
    }
});

function showResult(data) {
    document.getElementById('predictedDigit').textContent = data.digit;
    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1);
    
    const probBars = document.getElementById('probBars');
    probBars.innerHTML = '';
    
    data.probabilities.forEach((prob, idx) => {
        const barDiv = document.createElement('div');
        barDiv.className = 'prob-bar';
        barDiv.innerHTML = `
            <div class="prob-label">${idx}</div>
            <div class="prob-fill">
                <div class="prob-fill-inner" style="width: ${prob * 100}%"></div>
            </div>
            <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
        `;
        probBars.appendChild(barDiv);
    });
    
    resultDiv.classList.remove('hidden');
    errorDiv.classList.add('hidden');
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.add('hidden');
    resultDiv.classList.add('hidden');
}