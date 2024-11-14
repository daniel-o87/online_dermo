// components/LandingPage.tsx
import React from 'react';

export default function LandingPage() {
  return (
    <div>
      {/* Header Section */}
      <header className="bg-white w-full py-4 shadow-md text-center">
        <h2 className="text-2xl font-bold text-gray-800">Welcome to Melanoma Detection AI</h2>
      </header>

      {/* Hero Section */}
      <div
        className="hero h-screen bg-black flex flex-col items-center justify-center text-center text-white px-4"
        style={{ background: "url('/images/dermatology-bg.jpg') no-repeat center center/cover" }}
      >
        <div className="hero-content max-w-2xl bg-black bg-opacity-60 p-6 rounded-lg">
          <h1 className="text-5xl mb-6 text-white">Detect Melanoma Early with AI Technology</h1>
          <p className="text-lg mb-10 text-gray-300">
            Upload your image to get an instant analysis. Early detection is key to effective treatment. Our AI-driven
            tool helps you take a proactive step towards better health.
          </p>
          <button
            className="bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700"
            onClick={() => (window.location.href = '/upload')}
          >
            Get Started
          </button>
        </div>
      </div>

      {/* Picture Section Below Header */}
      <section className="w-full flex justify-center mt-6">
        <img
          src="/images/intro-image.jpg"
          alt="Dermatology Concept"
          className="w-full max-w-4xl rounded-lg shadow-md"
        />
      </section>
    </div>
  );
}
