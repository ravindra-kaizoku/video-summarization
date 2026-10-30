import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  UploadCloud,
  ChevronDown,
  Film,
  Music,
  Sparkles,
  Play,
  Clock,
  FileText,
  Sun,
  Moon,
  PanelLeft,
  Copy,
  Check,
  Download,
  AlertCircle,
  ExternalLink,
  RefreshCw,
  Video
} from 'lucide-react';
import './App.css';

const BACKEND_URL = (import.meta.env.VITE_API_URL || 'https://video-summarization-jwsl.onrender.com').replace(/\/+$/, '');
const API_BASE_URL = `${BACKEND_URL}/api`;

const YoutubeIcon = ({ size = 22, color = "#ef4444" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M22.54 6.42a2.78 2.78 0 0 0-1.94-2C18.88 4 12 4 12 4s-6.88 0-8.6.46a2.78 2.78 0 0 0-1.94 2A29 29 0 0 0 1 11.75a29 29 0 0 0 .46 5.33A2.78 2.78 0 0 0 3.4 19c1.72.46 8.6.46 8.6.46s6.88 0 8.6-.46a2.78 2.78 0 0 0 1.94-2 29 29 0 0 0 .46-5.25 29 29 0 0 0-.46-5.33z"></path>
    <polygon points="9.75 15.02 15.5 11.75 9.75 8.48 9.75 15.02"></polygon>
  </svg>
);

const TABS = [
  { label: 'YouTube Video', icon: <YoutubeIcon size={18} /> },
  { label: 'Upload Video', icon: <Film size={18} /> },
  { label: 'Upload Audio', icon: <Music size={18} /> },
  { label: 'Paste Transcript', icon: <FileText size={18} /> },
  { label: 'Direct Link', icon: <ExternalLink size={18} /> },
];

const QUICK_SAMPLES = [
  {
    title: 'Me at the zoo (19s Test)',
    url: 'https://www.youtube.com/watch?v=jNQXAC9IVRw',
    tag: 'Fast Test'
  },
  {
    title: 'Steve Jobs 2005 Speech',
    url: 'https://www.youtube.com/watch?v=UF8uR6Z6KLc',
    tag: '15 min'
  },
  {
    title: 'How AI Actually Works',
    url: 'https://www.youtube.com/watch?v=aircAruvnKk',
    tag: '10 min'
  }
];

const TEMPLATES = [
  { label: 'The Rise And Fall Of Spirit Airlines.mp4', type: 'VIDEO' },
  { label: 'The Real Reason Disney Is So Expensive.mp4', type: 'VIDEO' },
];

const MORE_TOOLS = [
  { id: 'youtube', label: 'YouTube Video Summarizer', icon: <YoutubeIcon size={22} color="#ef4444" /> },
  { id: 'transcript', label: 'Paste Transcript & Notes', icon: <FileText size={22} color="#10b981" /> },
  { id: 'audio', label: 'AI Audio Summarizer', icon: <Music size={22} color="#a855f7" /> },
  { id: 'video', label: 'Video to Text Converter', icon: <Film size={22} color="#3b82f6" /> },
];

function extractYoutubeId(url) {
  if (!url) return null;
  const match = url.match(/(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=|shorts\/))([\w-]{11})/);
  return match ? match[1] : null;
}

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState('YouTube Video');
  const [activeSection, setActiveSection] = useState('Templates');
  const [outputLanguage, setOutputLanguage] = useState('Auto');
  const [summaryTemplate, setSummaryTemplate] = useState('Chapter Summ...');

  const [file, setFile] = useState(null);
  const [videoUrlInput, setVideoUrlInput] = useState('');
  const [transcriptInput, setTranscriptInput] = useState('');
  const [transcriptTitle, setTranscriptTitle] = useState('');
  const [youtubeId, setYoutubeId] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);

  const [taskId, setTaskId] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);

  const [status, setStatus] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState(null);
  const [result, setResult] = useState(null);

  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [copied, setCopied] = useState(false);

  const videoRef = useRef(null);
  const fileInputRef = useRef(null);
  const urlInputRef = useRef(null);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  // Detect YouTube ID as user types
  useEffect(() => {
    const ytid = extractYoutubeId(videoUrlInput);
    setYoutubeId(ytid);

    if (ytid) {
      // Fast fallback preview
      setVideoPreview({
        title: 'YouTube Video Detected',
        thumbnail: `https://img.youtube.com/vi/${ytid}/hqdefault.jpg`,
        uploader: 'YouTube',
        duration_formatted: 'Detecting...'
      });

      // Fetch rich metadata
      const timer = setTimeout(async () => {
        try {
          setIsPreviewLoading(true);
          const resp = await axios.post(`${API_BASE_URL}/video-info`, { url: videoUrlInput.trim() });
          if (resp.data) {
            setVideoPreview(resp.data);
          }
        } catch (e) {
          console.warn('Preview fetch warning:', e);
        } finally {
          setIsPreviewLoading(false);
        }
      }, 500);

      return () => clearTimeout(timer);
    } else {
      setVideoPreview(null);
    }
  }, [videoUrlInput]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setVideoUrlInput('');
      setYoutubeId(null);
      setResult(null);
      setErrorMessage(null);
      uploadVideoFile(selectedFile);
    }
  };

  const uploadVideoFile = async (fileToUpload) => {
    setIsUploading(true);
    setStatus('uploading');
    setProgress(2);
    setStatusMessage('Preparing upload...');
    setErrorMessage(null);

    const sizeMB = (fileToUpload.size / (1024 * 1024)).toFixed(1);
    const CHUNK_SIZE = 10 * 1024 * 1024; // 10MB chunks (bypasses Cloudflare & Render 100MB body limits)

    // Fast direct upload for small files (<= 15MB)
    if (fileToUpload.size <= 15 * 1024 * 1024) {
      const formData = new FormData();
      formData.append('file', fileToUpload);
      try {
        const resp = await axios.post(`${API_BASE_URL}/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 120000,
          onUploadProgress: (progressEvent) => {
            if (progressEvent.total) {
              const pct = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              const loadedMB = (progressEvent.loaded / (1024 * 1024)).toFixed(1);
              const totalMB = (progressEvent.total / (1024 * 1024)).toFixed(1);
              if (pct >= 100) {
                setProgress(99);
                setStatusMessage(`Saving file on server (${totalMB} MB)...`);
              } else {
                setProgress(Math.min(99, Math.max(5, pct)));
                setStatusMessage(`Uploading: ${pct}% (${loadedMB} / ${totalMB} MB)...`);
              }
            }
          }
        });
        setTaskId(resp.data.task_id);
        setVideoUrl(`${BACKEND_URL}${resp.data.video_url}`);
        setStatus('uploaded');
        setProgress(100);
        setStatusMessage('Upload complete! Click "Start Summarizing" below.');
      } catch (err) {
        console.error('Upload error:', err);
        const msg = err.response?.data?.detail || err.message || 'Upload failed.';
        setStatusMessage(`Upload failed: ${msg}`);
        setErrorMessage(`Upload failed: ${msg}`);
      } finally {
        setIsUploading(false);
      }
      return;
    }

    // Chunked Upload for larger files (> 15MB, handles 100MB, 200MB, 500MB, 1GB+)
    const totalChunks = Math.ceil(fileToUpload.size / CHUNK_SIZE);
    const uploadTaskId = (typeof crypto !== 'undefined' && crypto.randomUUID)
      ? crypto.randomUUID()
      : 'task-' + Date.now() + '-' + Math.random().toString(36).substring(2, 9);

    try {
      let finalRespData = null;
      for (let i = 0; i < totalChunks; i++) {
        const start = i * CHUNK_SIZE;
        const end = Math.min(fileToUpload.size, start + CHUNK_SIZE);
        const chunkBlob = fileToUpload.slice(start, end);

        const formData = new FormData();
        formData.append('chunk', chunkBlob, fileToUpload.name);
        formData.append('task_id', uploadTaskId);
        formData.append('chunk_index', i);
        formData.append('total_chunks', totalChunks);
        formData.append('filename', fileToUpload.name);

        const resp = await axios.post(`${API_BASE_URL}/upload-chunk`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 180000,
          onUploadProgress: (progressEvent) => {
            const chunkLoaded = progressEvent.loaded || 0;
            const currentTotalLoaded = start + chunkLoaded;
            const overallPct = Math.min(99, Math.max(3, Math.round((currentTotalLoaded / fileToUpload.size) * 100)));
            const loadedMB = (currentTotalLoaded / (1024 * 1024)).toFixed(1);
            setProgress(overallPct);
            setStatusMessage(`Uploading: ${overallPct}% (${loadedMB} / ${sizeMB} MB) • Chunk ${i + 1}/${totalChunks}...`);
          }
        });

        if (i === totalChunks - 1) {
          finalRespData = resp.data;
        }
      }

      if (finalRespData && finalRespData.task_id) {
        setTaskId(finalRespData.task_id);
        if (finalRespData.video_url) {
          setVideoUrl(`${BACKEND_URL}${finalRespData.video_url}`);
        }
        setStatus('uploaded');
        setProgress(100);
        setStatusMessage(`Upload complete (${sizeMB} MB)! Click "Start Summarizing" below.`);
      }
    } catch (err) {
      console.error('Chunk upload error:', err);
      const msg = err.response?.data?.detail || err.message || 'Chunk upload failed.';
      setStatusMessage(`Upload failed: ${msg}`);
      setErrorMessage(`Upload failed: ${msg}. Please check your connection and try again.`);
    } finally {
      setIsUploading(false);
    }
  };

  const handlePasteClipboard = async () => {
    try {
      const text = await navigator.clipboard.readText();
      if (text) {
        setVideoUrlInput(text.trim());
      }
    } catch (err) {
      console.warn('Clipboard read error:', err);
    }
  };

  const handleSelectSample = (sampleUrl) => {
    setVideoUrlInput(sampleUrl);
  };

  // Submit Video URL (YouTube or direct video link)
  const handleUrlSubmit = async () => {
    if (!videoUrlInput.trim()) return;
    setErrorMessage(null);
    setResult(null);
    setIsProcessing(true);
    setStatus('processing');
    setProgress(5);
    setStatusMessage('Connecting to YouTube video stream...');

    try {
      const resp = await axios.post(`${API_BASE_URL}/process-url`, {
        url: videoUrlInput.trim()
      });
      setTaskId(resp.data.task_id);
      if (resp.data.youtube_id) {
        setYoutubeId(resp.data.youtube_id);
      }
    } catch (err) {
      console.error('URL process error:', err);
      const msg = err.response?.data?.detail || 'Failed to start video processing from URL.';
      setErrorMessage(msg);
      setStatusMessage('Failed to connect to video URL.');
      setIsProcessing(false);
    }
  };

  const handleTranscriptSubmit = async () => {
    if (!transcriptInput.trim()) return;
    setErrorMessage(null);
    setResult(null);
    setIsProcessing(true);
    setStatus('processing');
    setProgress(10);
    setStatusMessage('Analyzing and chunking transcript...');

    try {
      const resp = await axios.post(`${API_BASE_URL}/process-transcript`, {
        text: transcriptInput.trim(),
        title: transcriptTitle.trim() || 'Pasted Video Transcript'
      });
      setTaskId(resp.data.task_id);
    } catch (err) {
      console.error('Transcript process error:', err);
      const msg = err.response?.data?.detail || 'Failed to start transcript processing.';
      setErrorMessage(msg);
      setStatusMessage('Failed to process transcript.');
      setIsProcessing(false);
    }
  };

  const startSummarization = async () => {
    if (isUploading) {
      setStatusMessage('Please wait for the file upload to complete before starting summarization.');
      return;
    }
    if (!taskId) {
      setErrorMessage('No uploaded video found. Please re-select your file or check upload status.');
      return;
    }
    setErrorMessage(null);
    setIsProcessing(true);
    setStatus('processing');
    setProgress(5);
    setStatusMessage('Initiating multimodal summarization pipeline...');
    try {
      await axios.post(`${API_BASE_URL}/process/${taskId}`);
    } catch (err) {
      console.error('Process start error:', err);
      const msg = err.response?.data?.detail || 'Failed to start processing pipeline.';
      setStatusMessage('Failed to start processing.');
      setErrorMessage(msg);
      setIsProcessing(false);
    }
  };

  // Polling task status
  useEffect(() => {
    let interval = null;
    let failCount = 0;
    if (isProcessing && taskId) {
      interval = setInterval(async () => {
        try {
          const resp = await axios.get(`${API_BASE_URL}/status/${taskId}`);
          failCount = 0;
          setProgress(resp.data.progress || 0);
          setStatusMessage(resp.data.message || 'Processing video...');
          
          if (resp.data.video_url && !videoUrl) {
            setVideoUrl(`${BACKEND_URL}${resp.data.video_url}`);
          }
          if (resp.data.youtube_id && !youtubeId) {
            setYoutubeId(resp.data.youtube_id);
          }

          if (resp.data.status === 'completed') {
            setIsProcessing(false);
            clearInterval(interval);
            const resResp = await axios.get(`${API_BASE_URL}/result/${taskId}`);
            setResult(resResp.data);
            if (resResp.data.video_url) {
              setVideoUrl(`${BACKEND_URL}${resResp.data.video_url}`);
            }
          } else if (resp.data.status === 'failed') {
            setIsProcessing(false);
            clearInterval(interval);
            setErrorMessage(resp.data.message || resp.data.error || 'Video processing failed.');
          }
        } catch (err) {
          console.error('Status poll error:', err);
          failCount += 1;
          if (failCount >= 8) {
            setIsProcessing(false);
            clearInterval(interval);
            setErrorMessage('Lost connection to server. The server may have restarted due to memory limits. Try a shorter video or YouTube link.');
          }
        }
      }, 1500);
    }
    return () => clearInterval(interval);
  }, [isProcessing, taskId, videoUrl, youtubeId]);

  // Seek video player to exact timestamp
  const handleSeek = (seconds) => {
    if (youtubeId) {
      const iframe = document.getElementById('yt-player');
      if (iframe && iframe.contentWindow) {
        iframe.contentWindow.postMessage(
          JSON.stringify({
            event: 'command',
            func: 'seekTo',
            args: [seconds, true]
          }),
          '*'
        );
        iframe.contentWindow.postMessage(
          JSON.stringify({
            event: 'command',
            func: 'playVideo',
            args: []
          }),
          '*'
        );
      }
    } else if (videoRef.current) {
      videoRef.current.currentTime = seconds;
      videoRef.current.play();
    }
  };

  const handleCopySummary = () => {
    if (!result) return;
    const text = `VIDEO SUMMARY: ${result.title || file?.name || 'Video'}\n\n${result.summary}\n\nKEY TAKEAWAYS:\n${(result.key_points || []).map(p => `• ${p}`).join('\n')}`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadMarkdown = () => {
    if (!result) return;
    const content = `# Video Summary: ${result.title || file?.name || 'Video'}
Duration: ${result.video_duration || 'N/A'} | Chunks Processed: ${result.total_chunks_processed || 0}

## Executive Summary
${result.summary}

## Key Takeaways
${(result.key_points || []).map(p => `- ${p}`).join('\n')}

## Timestamped Chapters
${(result.important_sections || []).map(s => `### ${s.title} (${s.start} - ${s.end})\n${s.summary}\n`).join('\n')}
`;
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `video-summary-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleReset = () => {
    setTaskId(null);
    setResult(null);
    setVideoUrl(null);
    setFile(null);
    setVideoUrlInput('');
    setYoutubeId(null);
    setVideoPreview(null);
    setIsProcessing(false);
    setProgress(0);
    setStatusMessage('');
    setTranscriptInput('');
    setTranscriptTitle('');
    setErrorMessage(null);
  };

  const handleToolClick = (toolId) => {
    if (toolId === 'youtube') {
      setActiveTab('YouTube Video');
      if (urlInputRef.current) urlInputRef.current.focus();
    } else if (toolId === 'transcript') {
      setActiveTab('Paste Transcript');
    } else if (toolId === 'audio') {
      setActiveTab('Upload Audio');
    } else {
      setActiveTab('Upload Video');
    }
    window.scrollTo({ top: 150, behavior: 'smooth' });
  };

  return (
    <div className={`app-wrapper ${darkMode ? 'dark-mode' : ''}`}>
      {/* TOP HEADER */}
      <header className="top-header">
        <div className="header-inner">
          <div className="header-left">
            <button className="sidebar-icon-btn" aria-label="Sidebar Menu">
              <PanelLeft size={18} />
            </button>
            <div className="logo-area" onClick={handleReset} style={{ cursor: 'pointer' }}>
              <div className="logo-icon">
                <FileText size={18} />
              </div>
              <span className="logo-title">AI Video Summarizer</span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <button className="theme-toggle-btn" onClick={toggleDarkMode}>
              {darkMode ? <Sun size={16} /> : <Moon size={16} />}
              <span>{darkMode ? 'Light Mode' : 'Dark Mode'}</span>
            </button>
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* HERO */}
        <section className="hero-section">
          <h1 className="hero-title">
            Transform Any Video Into <span className="gradient-text">Instant AI Summaries</span>
          </h1>
          <p className="hero-subtitle">
            Paste any YouTube video link or upload media files to generate executive summaries, key takeaways & timestamped chapters with multimodal AI.
          </p>
        </section>

        {/* TABS */}
        <div className="tabs-scroll">
          <div className="tabs-container">
            {TABS.map((tab) => (
              <button
                key={tab.label}
                className={`tab-item ${activeTab === tab.label ? 'active' : ''}`}
                onClick={() => {
                  setActiveTab(tab.label);
                }}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* ERROR ALERT */}
        {errorMessage && (
          <div className="alert-error">
            <AlertCircle size={20} />
            <div style={{ flex: 1 }}>
              <strong>Processing Issue:</strong> {errorMessage}
              {(typeof errorMessage === 'string' && (errorMessage.toLowerCase().includes('blocking') || errorMessage.toLowerCase().includes('cloud') || errorMessage.toLowerCase().includes('transcript') || errorMessage.toLowerCase().includes('restrict'))) ? (
                <div style={{ marginTop: '10px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  <button
                    className="btn-retry"
                    style={{ background: '#3b82f6', color: 'white', borderColor: '#3b82f6', fontWeight: 'bold' }}
                    onClick={() => { setActiveTab('Paste Transcript'); setErrorMessage(null); }}
                  >
                    📝 Switch to Paste Transcript Tab
                  </button>
                  <button
                    className="btn-retry"
                    style={{ background: '#10b981', color: 'white', borderColor: '#10b981', fontWeight: 'bold' }}
                    onClick={() => { setActiveTab('Upload Video'); setErrorMessage(null); }}
                  >
                    📁 Switch to Upload Video File
                  </button>
                </div>
              ) : null}
            </div>
            <button className="btn-retry" onClick={handleReset}>
              <RefreshCw size={14} /> Try Again
            </button>
          </div>
        )}

        {/* MAIN INTERFACE CARD */}
        <div className="upload-card">
          {!result && !isProcessing && (
            <>
              {/* YOUTUBE VIDEO TAB */}
              {activeTab === 'YouTube Video' && (
                <div className="youtube-tab-content">
                  <div className="yt-badge-row">
                    <span className="yt-badge">
                      <YoutubeIcon size={16} /> YouTube URL Summarizer
                    </span>
                    <span className="yt-speed-badge">⚡ Ultra Fast 480p Stream</span>
                  </div>

                  <h2 className="upload-heading">Paste YouTube Video Link</h2>
                  <p className="upload-subtext">
                    Summarize lectures, podcasts, tutorials, documentaries or shorts instantly with multimodal AI.
                  </p>

                  <div className="youtube-input-box">
                    <div className="input-prefix-icon">
                      <YoutubeIcon size={24} color="#ef4444" />
                    </div>
                    <input
                      ref={urlInputRef}
                      type="text"
                      className="url-input yt-input"
                      placeholder="Paste YouTube link (e.g. https://www.youtube.com/watch?v=... or youtu.be/...)"
                      value={videoUrlInput}
                      onChange={(e) => setVideoUrlInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleUrlSubmit();
                      }}
                    />
                    <button className="btn-paste" onClick={handlePasteClipboard} title="Paste from clipboard">
                      Paste
                    </button>
                    <button
                      className="btn-upload yt-submit-btn"
                      disabled={!videoUrlInput.trim() || isProcessing}
                      onClick={handleUrlSubmit}
                    >
                      <Sparkles size={18} /> Summarize
                    </button>
                  </div>

                  {/* QUICK SAMPLE BUTTONS */}
                  <div className="quick-samples-row">
                    <span className="quick-samples-label">Try a sample video:</span>
                    {QUICK_SAMPLES.map((s, idx) => (
                      <button
                        key={idx}
                        className="sample-chip"
                        onClick={() => handleSelectSample(s.url)}
                      >
                        <Play size={11} style={{ fill: 'currentColor' }} />
                        <span>{s.title}</span>
                        <span className="sample-tag">{s.tag}</span>
                      </button>
                    ))}
                  </div>

                  {/* PREVIEW CARD */}
                  {videoPreview && (
                    <div className="video-preview-card">
                      <img src={videoPreview.thumbnail} alt="Video Thumbnail" className="preview-thumb" />
                      <div className="preview-details">
                        <div className="preview-title">{videoPreview.title}</div>
                        <div className="preview-meta">
                          <span>👤 {videoPreview.uploader}</span>
                          {videoPreview.duration_formatted && (
                            <span>⏱ {videoPreview.duration_formatted}</span>
                          )}
                          <span className="preview-ready-pill">Ready to summarize</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* UPLOAD LOCAL VIDEO TAB */}
              {activeTab === 'Upload Video' && (
                <div className="file-tab-content">
                  <h2 className="upload-heading">Upload Local Video File</h2>
                  <p className="upload-subtext">
                    Supports MP4, MOV, MKV, AVI files. Multimodal frame analysis & speech recognition will run locally.
                  </p>

                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="video/*"
                    style={{ display: 'none' }}
                  />

                  <div className="upload-actions">
                    <button
                      className="btn-upload"
                      onClick={() => fileInputRef.current.click()}
                      disabled={isUploading}
                    >
                      <UploadCloud size={20} />
                      {isUploading ? 'Uploading...' : 'Choose Video File'}
                    </button>
                  </div>

                  {isUploading && (
                    <div className="progress-card" style={{ marginTop: '1.25rem' }}>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                      </div>
                      <div className="progress-info">
                        <span>{statusMessage}</span>
                        <span className="prog-pct">{progress}%</span>
                      </div>
                    </div>
                  )}

                  {file && (
                    <div className="file-selected-badge" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem', marginTop: '1rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Film size={16} />
                        <span>Selected: <strong>{file.name}</strong> ({Math.round(file.size / 1024 / 1024)} MB)</span>
                      </div>
                      <button 
                        className="btn-upload" 
                        style={{ padding: '0.5rem 1.25rem', opacity: (isUploading || !taskId) ? 0.6 : 1 }} 
                        onClick={startSummarization}
                        disabled={isUploading || !taskId || isProcessing}
                      >
                        {isUploading ? (
                          <>
                            <RefreshCw size={14} className="spin-icon" /> Uploading ({progress}%)...
                          </>
                        ) : (
                          <>
                            <Sparkles size={16} /> Start Summarizing
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* UPLOAD AUDIO TAB */}
              {activeTab === 'Upload Audio' && (
                <div className="file-tab-content">
                  <h2 className="upload-heading">Upload Audio Recording</h2>
                  <p className="upload-subtext">
                    Transcribe and summarize podcasts, voice notes, lectures, and meetings in MP3, WAV, or M4A.
                  </p>

                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="audio/*"
                    style={{ display: 'none' }}
                  />

                  <div className="upload-actions">
                    <button
                      className="btn-upload"
                      onClick={() => fileInputRef.current.click()}
                      disabled={isUploading}
                    >
                      <Music size={20} />
                      {isUploading ? 'Uploading...' : 'Choose Audio File'}
                    </button>
                  </div>

                  {isUploading && (
                    <div className="progress-card" style={{ marginTop: '1.25rem' }}>
                      <div className="progress-track">
                        <div className="progress-fill" style={{ width: `${progress}%` }} />
                      </div>
                      <div className="progress-info">
                        <span>{statusMessage}</span>
                        <span className="prog-pct">{progress}%</span>
                      </div>
                    </div>
                  )}

                  {file && (
                    <div className="file-selected-badge" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem', marginTop: '1rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Music size={16} />
                        <span>Selected: <strong>{file.name}</strong></span>
                      </div>
                      <button 
                        className="btn-upload" 
                        style={{ padding: '0.5rem 1.25rem', opacity: (isUploading || !taskId) ? 0.6 : 1 }} 
                        onClick={startSummarization}
                        disabled={isUploading || !taskId || isProcessing}
                      >
                        {isUploading ? (
                          <>
                            <RefreshCw size={14} className="spin-icon" /> Uploading ({progress}%)...
                          </>
                        ) : (
                          <>
                            <Sparkles size={16} /> Start Summarizing
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              )}

              
              {/* PASTE TRANSCRIPT TAB */}
              {activeTab === 'Paste Transcript' && (
                <div className="youtube-tab-content">
                  <div className="yt-badge-row">
                    <span className="yt-badge">
                      <FileText size={16} /> Instant Transcript Summarizer
                    </span>
                    <span className="yt-pill">⚡ 100% Cloud-Proof (No YouTube IP Blocks)</span>
                  </div>

                  <h2 className="upload-heading">Paste Video Transcript or Spoken Notes</h2>
                  <p className="upload-subtext">
                    Paste YouTube captions, meeting transcripts, or lecture notes. Our AI generates an executive summary, distinct key takeaways, and timestamped chapter breakdown instantly without any cloud IP blocks.
                  </p>

                  <div style={{ margin: '18px 0' }}>
                    <input
                      type="text"
                      className="url-input yt-input"
                      style={{ marginBottom: '12px', width: '100%', boxSizing: 'border-box' }}
                      placeholder="Title or Topic (e.g. Will Harlow - 8 Strategies for Back Pain)"
                      value={transcriptTitle}
                      onChange={(e) => setTranscriptTitle(e.target.value)}
                    />
                    <textarea
                      className="url-input yt-input"
                      style={{
                        minHeight: '180px',
                        width: '100%',
                        boxSizing: 'border-box',
                        resize: 'vertical',
                        fontFamily: 'inherit',
                        fontSize: '14px',
                        lineHeight: '1.6',
                        padding: '12px 16px',
                        borderRadius: '10px'
                      }}
                      placeholder="Paste your transcript or subtitles here... (e.g. from YouTube 'Show transcript' or meeting notes)"
                      value={transcriptInput}
                      onChange={(e) => setTranscriptInput(e.target.value)}
                    />
                  </div>

                  <button
                    className="btn-upload yt-submit-btn"
                    disabled={!transcriptInput.trim() || isProcessing}
                    onClick={handleTranscriptSubmit}
                    style={{ width: '100%', justifyContent: 'center' }}
                  >
                    <Sparkles size={18} /> Generate Executive Summary & Key Takeaways
                  </button>
                </div>
              )}

              {/* DIRECT LINK TAB */}
              {activeTab === 'Direct Link' && (
                <div className="youtube-tab-content">
                  <h2 className="upload-heading">Paste Direct Video Link</h2>
                  <p className="upload-subtext">
                    Enter direct link to an MP4 video or hosted stream.
                  </p>
                  <div className="youtube-input-box">
                    <input
                      type="text"
                      className="url-input yt-input"
                      placeholder="Paste direct MP4 video link..."
                      value={videoUrlInput}
                      onChange={(e) => setVideoUrlInput(e.target.value)}
                    />
                    <button
                      className="btn-upload yt-submit-btn"
                      disabled={!videoUrlInput.trim() || isProcessing}
                      onClick={handleUrlSubmit}
                    >
                      <Sparkles size={18} /> Summarize
                    </button>
                  </div>
                </div>
              )}

              {/* SUMMARY OPTIONS */}
              <div className="options-row">
                <div className="option-group">
                  <label className="option-label">Output Language:</label>
                  <div className="select-wrapper">
                    <select
                      className="custom-select"
                      value={outputLanguage}
                      onChange={(e) => setOutputLanguage(e.target.value)}
                    >
                      <option>Auto</option>
                      <option>English</option>
                      <option>Hindi</option>
                      <option>Spanish</option>
                      <option>French</option>
                    </select>
                    <ChevronDown size={14} className="select-chevron" />
                  </div>
                </div>

                <div className="option-group">
                  <label className="option-label">Summary Format:</label>
                  <div className="select-wrapper">
                    <select
                      className="custom-select"
                      value={summaryTemplate}
                      onChange={(e) => setSummaryTemplate(e.target.value)}
                    >
                      <option>Chapter Summaries</option>
                      <option>Key Points</option>
                      <option>Executive Brief</option>
                      <option>Detailed Notes</option>
                    </select>
                    <ChevronDown size={14} className="select-chevron" />
                  </div>
                </div>
              </div>
            </>
          )}

          {/* PROCESSING STATE WITH ANIMATED STAGES */}
          {isProcessing && (
            <div className="processing-view">
              <div className="processing-header">
                <div className="pulsing-spinner">
                  <Sparkles size={28} className="spin-icon" />
                </div>
                <h3>Processing Video with Multimodal AI</h3>
                <p className="processing-submsg">{statusMessage}</p>
              </div>

              {/* VIDEO EMBED PREVIEW DURING PROCESSING */}
              {youtubeId && (
                <div className="processing-yt-preview">
                  <iframe
                    id="yt-player-preview"
                    className="yt-iframe-preview"
                    src={`https://www.youtube.com/embed/${youtubeId}?autoplay=0&mute=1`}
                    title="YouTube video player"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                </div>
              )}

              {/* PROGRESS BAR */}
              <div className="progress-card">
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${Math.max(8, progress)}%` }} />
                </div>
                <div className="progress-info">
                  <span>{statusMessage}</span>
                  <span className="prog-pct">{progress}%</span>
                </div>
              </div>

              {/* PIPELINE MILESTONE CHECKLIST */}
              <div className="pipeline-steps-grid">
                <div className={`step-item ${progress >= 10 ? 'active' : ''} ${progress >= 25 ? 'done' : ''}`}>
                  <span className="step-dot">{progress >= 25 ? '✓' : '1'}</span>
                  <span>Fetch & Stream Video</span>
                </div>
                <div className={`step-item ${progress >= 25 ? 'active' : ''} ${progress >= 50 ? 'done' : ''}`}>
                  <span className="step-dot">{progress >= 50 ? '✓' : '2'}</span>
                  <span>Whisper Speech STT</span>
                </div>
                <div className={`step-item ${progress >= 50 ? 'active' : ''} ${progress >= 80 ? 'done' : ''}`}>
                  <span className="step-dot">{progress >= 80 ? '✓' : '3'}</span>
                  <span>Visual Frame Analysis</span>
                </div>
                <div className={`step-item ${progress >= 80 ? 'active' : ''} ${progress >= 90 ? 'done' : ''}`}>
                  <span className="step-dot">{progress >= 90 ? '✓' : '4'}</span>
                  <span>PyTorch Segment Scoring</span>
                </div>
                <div className={`step-item ${progress >= 90 ? 'active' : ''} ${progress >= 100 ? 'done' : ''}`}>
                  <span className="step-dot">{progress >= 100 ? '✓' : '5'}</span>
                  <span>Hierarchical LLM Summary</span>
                </div>
              </div>
            </div>
          )}

          {/* READY VIDEO PLAYER WHEN PROCESSING COMPLETE OR UPLOADED */}
          {result && (
            <div className="completed-player-header">
              <div className="result-top-bar">
                <div className="result-title-group">
                  <span className="source-tag">
                    {youtubeId ? <YoutubeIcon size={16} /> : <Film size={16} />}
                    {youtubeId ? 'YouTube Stream' : 'Uploaded Video'}
                  </span>
                  <h3 className="result-title">{result.title || file?.name || 'Summarized Video'}</h3>
                </div>
                <div className="result-action-btns">
                  <button className="btn-action-pill" onClick={handleCopySummary}>
                    {copied ? <Check size={15} color="#10b981" /> : <Copy size={15} />}
                    <span>{copied ? 'Copied!' : 'Copy Summary'}</span>
                  </button>
                  <button className="btn-action-pill" onClick={handleDownloadMarkdown}>
                    <Download size={15} />
                    <span>Export MD</span>
                  </button>
                  <button className="btn-action-pill" onClick={handleReset}>
                    <RefreshCw size={15} />
                    <span>New Video</span>
                  </button>
                </div>
              </div>

              {/* INTEGRATED INTERACTIVE PLAYER */}
              <div className="video-player-container">
                {youtubeId ? (
                  <iframe
                    id="yt-player"
                    className="yt-iframe-main"
                    src={`https://www.youtube.com/embed/${youtubeId}?enablejsapi=1&origin=${window.location.origin}`}
                    title="YouTube video player"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                  />
                ) : videoUrl ? (
                  <video ref={videoRef} src={videoUrl} controls className="video-element" />
                ) : null}
              </div>
            </div>
          )}
        </div>

        {/* RESULTS SECTION */}
        {result && (
          <div className="results-wrapper">
            {/* EXECUTIVE SUMMARY */}
            <div className="summary-card main-summary-card">
              <div className="summary-card-header">
                <h2><Sparkles size={22} className="sparkle-gold" /> Executive Video Summary</h2>
                <div className="summary-meta-chips">
                  <span className="meta-chip">⏱ Duration: <strong>{result.video_duration}</strong></span>
                  <span className="meta-chip">🧩 Chunks: <strong>{result.total_chunks_processed}</strong></span>
                  <span className="meta-chip">⭐ Top Moments: <strong>{result.selected_chunks_count}</strong></span>
                </div>
              </div>
              <p className="executive-summary-text">
                {result.summary}
              </p>
            </div>

            {/* TWO COLUMN GRID: KEY POINTS & TIMESTAMPED CHAPTERS */}
            <div className="key-points-grid">
              <div className="summary-card">
                <h2>📌 Key Takeaways</h2>
                <div className="points-container">
                  {result.key_points && result.key_points.map((pt, idx) => (
                    <div key={idx} className="point-item">
                      <div className="point-num">{idx + 1}</div>
                      <div className="point-text">{pt}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="summary-card">
                <h2>⏱ Timestamped Chapters</h2>
                <p className="chapter-help-text">Click any chapter to jump the player directly to that moment.</p>
                <div className="chapters-container">
                  {result.important_sections && result.important_sections.map((sec, idx) => (
                    <div key={idx} className="timestamp-section">
                      <div className="timestamp-header">
                        <span className="timestamp-badge">{sec.start} - {sec.end}</span>
                        <button className="timestamp-seek-btn" onClick={() => handleSeek(sec.start_seconds)}>
                          <Play size={13} style={{ display: 'inline', fill: 'currentColor' }} /> Jump to {sec.start}
                        </button>
                      </div>
                      <div className="timestamp-title">{sec.title}</div>
                      <p className="timestamp-summary">{sec.summary}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TEMPLATES / HISTORY (WHEN IDLE) */}
        {!result && !isProcessing && (
          <div className="section-panel">
            <div className="section-tabs">
              {['Templates', 'History'].map((s) => (
                <button
                  key={s}
                  className={`section-tab ${activeSection === s ? 'active' : ''}`}
                  onClick={() => setActiveSection(s)}
                >
                  {s === 'History' && <Clock size={14} style={{ marginRight: '4px' }} />}
                  {s}
                </button>
              ))}
            </div>

            <div className="template-grid">
              {TEMPLATES.map((t, i) => (
                <div className="template-card" key={i}>
                  <div className="template-icon">
                    <Film size={24} />
                  </div>
                  <div>
                    <div className="template-type">{t.type}</div>
                    <div className="template-name">{t.label}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* MORE TOOLS */}
        <div className="more-tools-section">
          <h3 className="more-tools-heading">Explore AI Media Suite</h3>
          <div className="more-tools-grid">
            {MORE_TOOLS.map((tool) => (
              <div className="tool-card" key={tool.id} onClick={() => handleToolClick(tool.id)}>
                <div className="tool-icon">{tool.icon}</div>
                <div>
                  <div className="tool-label">{tool.label}</div>
                  <div className="tool-sublabel">Click to launch</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
