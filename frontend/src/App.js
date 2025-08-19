import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        setError(err.error || "Server error");
        setLoading(false);
        return;
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Could not connect to backend.");
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 500, margin: "40px auto", fontFamily: "Arial" }}>
      <h2>Fashion Visual Search</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ marginBottom: 10 }}
        />
        <br />
        <button type="submit" disabled={loading}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && (
        <div style={{ color: "red", marginTop: 20 }}>
          <b>Error:</b> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 20 }}>
          <h4>Results:</h4>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;

