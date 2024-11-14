// components/MLInterface.tsx
'use client';
import React, { useState } from 'react';
import { Upload, Loader } from 'lucide-react';

interface Metadata {
  sex: string;
  anatom_site_general_challenge: string;
  age_approx: number | string;
}

interface Prediction {
  prediction: number;
  prediction_probability: number;
  processing_time_ms: number;
}

export default function MLInterface() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [metadata, setMetadata] = useState<Metadata>({
    sex: 'male',  // Default value
    anatom_site_general_challenge: 'torso',  // Default value
    age_approx: 70  // Default value
  });

  const handlePredict = async () => {
    if (!file) {
      setError('Please select a file to proceed with analysis.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('metadata', JSON.stringify(metadata));

      const response = await fetch('http://localhost:8008/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      setError('Unable to process the image. Ensure the file format is supported and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-8 bg-gray-50">
      <div className="bg-white rounded-xl shadow-lg p-8">
        <h1 className="text-3xl font-semibold mb-8 text-gray-800">
          Melanoma Detection
        </h1>

        {/* Step Guide */}
        <p className="text-sm text-gray-600 mb-4">
          Follow these steps: 1. Upload an Image 2. Provide Metadata 3. Click 'Get Prediction'
        </p>

        {/* File Upload with Drag-and-Drop */}
        <div
          className="border-2 border-dashed border-gray-200 rounded-xl p-8 hover:border-blue-500 transition-all duration-200 cursor-pointer"
          onClick={() => document.getElementById('file-upload')?.click()}
        >
          <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <p className="text-center text-gray-800">
            {file ? `Selected File: ${file.name}` : 'Drag and drop a DICOM file here or click to upload'}
          </p>
          <input
            id="file-upload"
            type="file"
            className="hidden"
            onChange={(e) => e.target.files && setFile(e.target.files[0])}
            accept=".dcm"
          />
        </div>

        {/* Metadata Form */}
        <div className="mt-6 space-y-4">
          <h3 className="font-medium text-gray-800">Patient Information</h3>
          <div className="grid grid-cols-2 gap-4">
            {/* Sex Selection */}
            <div>
              <label className="block mb-2 text-sm font-medium text-gray-900">
                Sex
              </label>
              <select
                className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg 
                          focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5
                          disabled:bg-gray-100 disabled:text-gray-500"
                value={metadata.sex}
                onChange={(e) => setMetadata({ ...metadata, sex: e.target.value })}
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>

            {/* Age Input */}
            <div>
              <label className="block mb-2 text-sm font-medium text-gray-900">
                Age
              </label>
              <input
                type="number"
                className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg 
                          focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5
                          placeholder:text-gray-400"
                value={metadata.age_approx}
                onChange={(e) => setMetadata({
                  ...metadata,
                  age_approx: parseFloat(e.target.value),
                })}
                placeholder="Enter age"
              />
            </div>

            {/* Anatomic Site Selection */}
            <div className="col-span-2">
              <label className="block mb-2 text-sm font-medium text-gray-900">
                Anatomic Site
              </label>
              <select
                className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg 
                          focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5
                          disabled:bg-gray-100 disabled:text-gray-500"
                value={metadata.anatom_site_general_challenge}
                onChange={(e) => setMetadata({
                  ...metadata,
                  anatom_site_general_challenge: e.target.value,
                })}
              >
                <option value="torso">Torso</option>
                <option value="upper extremity">Upper Extremity</option>
                <option value="lower extremity">Lower Extremity</option>
                <option value="head/neck">Head/Neck</option>
              </select>
            </div>
          </div>
        </div>

        {/* Loading Indicator */}
        {loading && (
          <div className="flex justify-center mt-6">
            <Loader className="animate-spin w-8 h-8 text-blue-500" />
          </div>
        )}

        {/* Submit Button */}
        <button
          className="w-full mt-6 bg-blue-500 text-white py-2 px-4 rounded-lg 
                      hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed
                      transition-colors duration-200"
          disabled={!file || loading}
          onClick={handlePredict}
        >
          {loading ? 'Analyzing...' : 'Get Prediction'}
        </button>
        {/* Results */}
        {prediction && (
          <div className="mt-6 bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold mb-4 text-gray-800">Analysis Results:</h3>
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <span className="text-gray-600">Prediction:</span>
                <span className="text-gray-800 font-medium">
                  {prediction.prediction === 1 ? 'Positive' : 'Negative'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <span className="text-gray-600">Probability:</span>
                <span className="text-gray-800 font-medium">
                  {(prediction.prediction_probability * 100).toFixed(2)}%
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <span className="text-gray-600">Processing Time:</span>
                <span className="text-gray-800 font-medium">
                  {prediction.processing_time_ms.toFixed(2)}ms
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
