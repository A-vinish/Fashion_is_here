import { useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { UploadCloud, ImageIcon, X } from "lucide-react";

const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];

export default function UploadDropzone({ onFileSelected, isBusy }) {
  const inputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState("");

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      alert("Please upload a PNG, JPG, JPEG, or WEBP image.");
      return;
    }
    setPreview(URL.createObjectURL(file));
    setFileName(file.name);
    onFileSelected(file);
  }, [onFileSelected]);

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  }

  function clearPreview(e) {
    e.stopPropagation();
    setPreview(null);
    setFileName("");
    onFileSelected(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label="Upload a fashion image by clicking or dragging a file here"
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={`relative cursor-pointer rounded-3xl border-2 border-dashed p-8 text-center
                  transition-colors duration-200 bg-white/70 backdrop-blur-sm shadow-card
                  ${isDragging ? "border-primary bg-primary/5" : "border-secondary/30 hover:border-secondary/60"}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      <AnimatePresence mode="wait">
        {preview ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            className="relative inline-block"
          >
            <img
              src={preview}
              alt="Selected upload preview"
              className="max-h-52 rounded-2xl object-cover shadow-card"
            />
            <button
              onClick={clearPreview}
              aria-label="Remove selected image"
              className="absolute -top-2 -right-2 w-7 h-7 rounded-full bg-ink text-white flex items-center justify-center shadow-card"
            >
              <X size={14} />
            </button>
            <p className="mt-3 text-sm text-muted flex items-center justify-center gap-1.5">
              <ImageIcon size={14} /> {fileName}
            </p>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-3 py-6"
          >
            <motion.div
              animate={{ y: [0, -6, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              className="w-14 h-14 rounded-2xl bg-gradient-to-br from-primary to-secondary
                         flex items-center justify-center text-white shadow-glow"
            >
              <UploadCloud size={24} />
            </motion.div>
            <p className="font-medium text-ink">Drag & drop a fashion photo here</p>
            <p className="text-sm text-muted">or click to browse — PNG, JPG, JPEG, WEBP</p>
          </motion.div>
        )}
      </AnimatePresence>

      {isBusy && (
        <div className="absolute bottom-0 left-0 right-0 h-1 overflow-hidden rounded-b-3xl">
          <div className="h-full w-1/3 bg-gradient-to-r from-primary to-accent animate-[shimmer_1.2s_linear_infinite]" />
        </div>
      )}
    </div>
  );
}
