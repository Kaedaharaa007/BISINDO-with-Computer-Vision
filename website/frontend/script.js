
const video = document.getElementById("webcam");

const predictionText = document.getElementById("prediction");
const confidenceText = document.getElementById("confidence");
const statusText = document.getElementById("status");

const startButton = document.getElementById("startButton");
const stopButton = document.getElementById("stopButton");

const API_URL = "http://localhost:8000/predict";

let webcamStream = null;
let isWebcamRunning = false;
let isProcessing = false;
let lastPredictionTime = 0;

// Temporary canvas for capturing webcam frames
const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d");

async function startWebcam() {

    try {

        if (isWebcamRunning) return;

        statusText.textContent = "Starting webcam...";

        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480
            }
        });

        video.srcObject = webcamStream;

        await video.play();

        isWebcamRunning = true;

        startButton.disabled = true;
        stopButton.disabled = false;

        statusText.textContent = "Camera running";

        requestAnimationFrame(processFrame);

    } catch (error) {

        console.error(error);

        statusText.textContent = error.message;

        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }

        video.srcObject = null;
    }
}

function stopWebcam() {

    isWebcamRunning = false;

    if (webcamStream) {

        webcamStream.getTracks().forEach(track => {
            track.stop();
        });

        webcamStream = null;
    }

    video.srcObject = null;

    predictionText.textContent = "Waiting...";
    confidenceText.textContent = "Confidence: --";

    statusText.textContent = "Camera stopped";

    startButton.disabled = false;
    stopButton.disabled = true;
}

async function sendFrame() {

    // Match the webcam dimensions
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;

    // Capture current video frame
    captureContext.drawImage(
        video,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height
    );

    // Convert frame to JPEG
    const blob = await new Promise(resolve => {

        captureCanvas.toBlob(
            resolve,
            "image/jpeg",
            0.85
        );

    });

    if (!blob || !isWebcamRunning) return;

    // Prepare image upload
    const formData = new FormData();

    formData.append(
        "file",
        blob,
        "frame.jpg"
    );

    // Send frame to FastAPI
    const response = await fetch(API_URL, {

        method: "POST",
        body: formData

    });

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    const result = await response.json();

    // Ignore response if webcam was stopped
    if (!isWebcamRunning) return;

    if (!result.success) {

        statusText.textContent = result.message;

        return;
    }

    // No hand detected
    if (!result.detected) {

        predictionText.textContent = "--";
        confidenceText.textContent = "No hand detected";
        statusText.textContent = "Show your hand to the camera";

        return;
    }

    // Display prediction
    predictionText.textContent = result.prediction;

    confidenceText.textContent =
        `Confidence: ${(result.confidence * 100).toFixed(2)}%`;

    statusText.textContent =
        `Detected ${result.hands_detected} hand(s)`;
}

async function processFrame(timestamp) {

    if (!isWebcamRunning) return;

    // Process approximately 5 frames per second
    if (
        timestamp - lastPredictionTime >= 200 &&
        !isProcessing &&
        video.readyState >= 2
    ) {

        isProcessing = true;

        lastPredictionTime = timestamp;

        try {

            await sendFrame();

        } catch (error) {

            console.error(error);

            if (isWebcamRunning) {
                statusText.textContent = "API connection error";
            }

        } finally {

            isProcessing = false;

        }
    }

    // Continue processing
    if (isWebcamRunning) {
        requestAnimationFrame(processFrame);
    }
}

startButton.addEventListener("click", startWebcam);
stopButton.addEventListener("click", stopWebcam);
stopButton.disabled = true;
