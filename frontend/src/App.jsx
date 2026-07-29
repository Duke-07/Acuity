import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { ReactCompareSlider, ReactCompareSliderImage } from 'react-compare-slider';
import { UploadCloud, Download, Loader2 } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  
  const [model, setModel] = useState('realesrgan');
  const [scale, setScale] = useState(4);
  const [faceEnhance, setFaceEnhance] = useState(false);
  
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle');
  const [resultUrl, setResultUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  const onDrop = useCallback(acceptedFiles => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setPreviewUrl(URL.createObjectURL(acceptedFiles[0]));
      setStatus('idle');
      setJobId(null);
      setResultUrl(null);
      setErrorMsg('');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/webp': [] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024
  });

  const handleUpscale = async () => {
    if (!file) return;
    
    setStatus('uploading');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', model);
    formData.append('scale', scale);
    formData.append('face_enhance', faceEnhance);

    try {
      const response = await fetch(`${API_BASE}/api/upscale`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Upload failed');
      }
      
      const data = await response.json();
      setJobId(data.job_id);
      setStatus('queued');
    } catch (err) {
      setStatus('failed');
      setErrorMsg(err.message);
    }
  };

  useEffect(() => {
    if (!jobId || status === 'done' || status === 'failed') return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        if (!res.ok) throw new Error('Failed to fetch job status');
        const data = await res.json();
        
        setStatus(data.status);
        if (data.status === 'done') {
          setResultUrl(`${API_BASE}${data.result_url}`);
          clearInterval(interval);
        } else if (data.status === 'failed') {
          setErrorMsg(data.error || 'Upscaling failed');
          clearInterval(interval);
        }
      } catch (err) {
        console.error(err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [jobId, status]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-8 font-sans">
      <header className="mb-14 text-center">
        <h1 className="text-5xl font-extrabold tracking-tighter text-white mb-3 bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent drop-shadow-2xl">
          Acuity
        </h1>
        <p className="text-gray-400 text-lg tracking-wide font-light">High-Fidelity AI Image Upscaling</p>
      </header>

      <main className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-3 gap-8">
        
        <div className="md:col-span-1 bg-gray-800/60 backdrop-blur-xl rounded-2xl p-6 shadow-[0_0_40px_rgba(139,92,246,0.15)] border border-gray-700/50">
          
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">AI Model</label>
            <select 
              value={model} 
              onChange={e => setModel(e.target.value)}
              className="w-full bg-gray-900/80 border border-gray-600 rounded-lg px-3 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-shadow"
            >
              <option value="realesrgan">Real-ESRGAN (General)</option>
              <option value="realesrgan_anime">Real-ESRGAN (Anime)</option>
              <option value="swinir">SwinIR (Real-World)</option>
            </select>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">Scale Factor</label>
            <div className="flex gap-2">
              {[2, 4].map(s => (
                <button
                  key={s}
                  onClick={() => setScale(s)}
                  className={`flex-1 py-2 rounded-lg font-medium transition-colors ${
                    scale === s ? 'bg-purple-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          </div>

          <div className="mb-8">
            <label className="flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={faceEnhance}
                onChange={e => setFaceEnhance(e.target.checked)}
                className="w-5 h-5 rounded border-gray-600 text-purple-600 focus:ring-purple-500 bg-gray-700"
              />
              <span className="text-sm font-medium text-gray-300">Face Enhancement (GFPGAN)</span>
            </label>
          </div>

          <button
            onClick={handleUpscale}
            disabled={!file || status === 'uploading' || status === 'queued' || status === 'processing'}
            className="w-full py-3.5 rounded-xl font-bold text-white bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-[0_0_20px_rgba(139,92,246,0.4)] hover:shadow-[0_0_30px_rgba(139,92,246,0.6)] flex justify-center items-center gap-2 tracking-wide"
          >
            {status === 'uploading' && <Loader2 className="animate-spin" size={20} />}
            {status === 'queued' && <Loader2 className="animate-spin" size={20} />}
            {status === 'processing' && <Loader2 className="animate-spin" size={20} />}
            {['idle', 'done', 'failed'].includes(status) && 'Upscale Image'}
            {['uploading', 'queued', 'processing'].includes(status) && status.charAt(0).toUpperCase() + status.slice(1)}
          </button>
          
          {errorMsg && (
            <div className="mt-4 p-3 bg-red-900/50 border border-red-500/50 rounded-lg text-red-200 text-sm">
              {errorMsg}
            </div>
          )}
        </div>

        <div className="md:col-span-2 flex flex-col gap-4">
          
          {!resultUrl ? (
            <div 
              {...getRootProps()} 
              className={`flex flex-col items-center justify-center w-full h-[500px] border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer overflow-hidden ${
                isDragActive ? 'border-cyan-400 bg-cyan-500/10 shadow-[0_0_30px_rgba(34,211,238,0.2)]' : 'border-gray-600 bg-gray-800/40 hover:bg-gray-800/60 hover:border-purple-500/50'
              }`}
            >
              <input {...getInputProps()} />
              
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" className="w-full h-full object-contain p-2" />
              ) : (
                <div className="text-center p-6">
                  <UploadCloud className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-gray-300 mb-2">Drag & drop an image here</p>
                  <p className="text-sm text-gray-500">Max 10MB, up to 4000x4000px</p>
                </div>
              )}
            </div>
          ) : (
            <div className="w-full h-[500px] rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(0,0,0,0.5)] border border-gray-700/80 relative">
              <ReactCompareSlider
                itemOne={<ReactCompareSliderImage src={previewUrl} alt="Original" />}
                itemTwo={<ReactCompareSliderImage src={resultUrl} alt="Upscaled" />}
                className="w-full h-full"
              />
              <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm text-white px-3 py-1 rounded-full text-xs font-medium tracking-wide z-10 pointer-events-none">Before</div>
              <div className="absolute top-4 right-4 bg-black/60 backdrop-blur-sm text-white px-3 py-1 rounded-full text-xs font-medium tracking-wide z-10 pointer-events-none">After</div>
            </div>
          )}

          {resultUrl && (
            <div className="flex justify-end">
              <a 
                href={resultUrl} 
                download
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-gray-800 to-gray-700 hover:from-gray-700 hover:to-gray-600 text-white font-medium rounded-xl transition-all border border-gray-600 shadow-[0_0_15px_rgba(255,255,255,0.05)] hover:shadow-[0_0_20px_rgba(255,255,255,0.1)]"
              >
                <Download size={18} />
                Download Result
              </a>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
