import React, { useState, useRef } from 'react';
import { Mic, Square, Loader2, Play, Volume2, Upload, Send, FileText } from 'lucide-react';

export default function App() {
    // Mode states
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [audioURL, setAudioURL] = useState(null);
    const [error, setError] = useState(null);

    // Text Mode states
    const [textQuery, setTextQuery] = useState("");
    const [chatResponse, setChatResponse] = useState("");

    // PDF Upload states
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadStatus, setUploadStatus] = useState("");
    const [isUploading, setIsUploading] = useState(false);

    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    // --- Voice Mode Logic ---
    const startRecording = async () => {
        setError(null);
        setChatResponse("");
        setAudioURL(null);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorderRef.current.onstop = handleStop;
            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (err) {
            console.error("Error accessing microphone:", err);
            alert("Microphone access denied or not available.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
        }
    };

    const handleStop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        sendVoiceToBackend(audioBlob);
    };

    const sendVoiceToBackend = async (blob) => {
        setIsProcessing(true);
        const formData = new FormData();
        formData.append('file', blob, 'recording.wav');

        try {
            const response = await fetch('/api/voice-query', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || "Voice query failed");
            }

            const returnedBlob = await response.blob();
            const url = URL.createObjectURL(returnedBlob);
            setAudioURL(url);
            new Audio(url).play();
        } catch (err) {
            setError(err.message);
        } finally {
            setIsProcessing(false);
        }
    };

    // --- Text Mode Logic ---
    const handleTextSubmit = async (e) => {
        e.preventDefault();
        if (!textQuery.trim()) return;

        setIsProcessing(true);
        setError(null);
        setAudioURL(null);
        setChatResponse("");

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: textQuery }),
            });

            if (!response.ok) {
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.includes("application/json")) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || "Chat request failed");
                } else {
                    const errorText = await response.text();
                    throw new Error(errorText || `Server error (${response.status})`);
                }
            }

            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                const data = await response.json();
                setChatResponse(data.response);
                setTextQuery("");
            } else {
                throw new Error("Server returned non-JSON response. Is the backend running?");
            }
        } catch (err) {
            setError(err.message === "Failed to fetch" ? "Cannot connect to backend server. Make sure it is running on port 8000." : err.message);
        } finally {
            setIsProcessing(false);
        }
    };

    // --- PDF Upload Logic ---
    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
        setUploadStatus("");
    };

    const handleUpload = async () => {
        if (!selectedFile) return;

        setIsUploading(true);
        setUploadStatus("Uploading...");
        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/api/upload-pdf', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const contentType = response.headers.get("content-type");
                if (contentType && contentType.includes("application/json")) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || errorData.message || "Upload failed");
                } else {
                    const text = await response.text();
                    throw new Error(text || `Server error (${response.status})`);
                }
            }

            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                const data = await response.json();
                setUploadStatus(data.message || "Success! PDF Indexed.");
            } else {
                setUploadStatus("Success! PDF Indexed.");
            }

            setSelectedFile(null);
            // Reset the file input
            const fileInput = document.getElementById('pdf-upload');
            if (fileInput) fileInput.value = "";
        } catch (err) {
            setUploadStatus(`Error: ${err.message === "Failed to fetch" ? "Cannot connect to server." : err.message}`);
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="container">
            <header>
                <h1>AI RAG <span>Assistant</span></h1>
                <p>Knowledge Base & Voice interaction</p>
            </header>

            <main>
                {/* PDF Upload Section */}
                <div className="card upload-section">
                    <h3><FileText size={18} /> Upload Knowledge (PDF)</h3>
                    <div className="upload-controls">
                        <input
                            type="file"
                            id="pdf-upload"
                            accept=".pdf"
                            onChange={handleFileChange}
                            disabled={isUploading}
                        />
                        <button
                            className="btn btn-upload"
                            onClick={handleUpload}
                            disabled={!selectedFile || isUploading}
                        >
                            {isUploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
                            Upload PDF
                        </button>
                    </div>
                    {uploadStatus && (
                        <div className={`status-msg ${uploadStatus.includes('Error') ? 'err' : 'success'}`}>
                            {uploadStatus}
                        </div>
                    )}
                </div>

                {/* Query Section */}
                <div className="card query-section">
                    <div className="status-indicator">
                        {isRecording ? (
                            <div className="pulse-red">Recording Voice...</div>
                        ) : isProcessing ? (
                            <div className="pulse-blue">AI is Thinking...</div>
                        ) : (
                            <div className="idle">Ask me anything</div>
                        )}
                    </div>

                    {/* Chat Response Display */}
                    {chatResponse && (
                        <div className="chat-bubble">
                            <div className="bubble-label">AI Response:</div>
                            <div className="bubble-content">{chatResponse}</div>
                        </div>
                    )}

                    {/* Audio Feedback */}
                    {audioURL && !isProcessing && (
                        <div className="playback">
                            <Volume2 size={20} />
                            <span>Voice Response Playing</span>
                            <button className="btn-play" onClick={() => new Audio(audioURL).play()}>
                                <Play size={16} /> Replay
                            </button>
                        </div>
                    )}

                    {error && (
                        <div className="error-display">
                            ⚠️ Error: {error}
                        </div>
                    )}

                    {/* Multi-Mode Inputs */}
                    <div className="input-modes">
                        <form onSubmit={handleTextSubmit} className="text-input-form">
                            <input
                                type="text"
                                value={textQuery}
                                onChange={(e) => setTextQuery(e.target.value)}
                                placeholder="Type your question..."
                                disabled={isProcessing || isRecording}
                            />
                            <button type="submit" className="btn btn-send" disabled={!textQuery || isProcessing || isRecording}>
                                <Send size={20} />
                            </button>
                        </form>

                        <div className="divider">OR</div>

                        <div className="btn-group">
                            {!isRecording ? (
                                <button
                                    className="btn btn-record"
                                    onClick={startRecording}
                                    disabled={isProcessing}
                                >
                                    <Mic size={24} />
                                    Ask by Voice
                                </button>
                            ) : (
                                <button className="btn btn-stop" onClick={stopRecording}>
                                    <Square size={24} />
                                    Stop & Process
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </main>

            <footer>
                <p>CPU Optimized • Whisper • Llama 3 • Coqui TTS</p>
            </footer>
        </div>
    );
}
