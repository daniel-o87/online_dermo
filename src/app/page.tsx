import React from 'react';

export default function Home(): JSX.Element {
  return (
    <main className="min-h-screen bg-white flex flex-col justify-center items-center w-full">
      {/* Header */}
      <header className="bg-white py-6 w-full">
        <div className="flex justify-end w-full px-4">
          <h1 className="text-4xl font-bold text-black">Melanoma Detection</h1>
        </div>
      </header>
      
      {/* Image */}
      <div className="flex justify-center w-full px-4 py-8">
        <img
          src="/images/homepage.jpg"
          alt="Homepage"
          className="w-auto h-auto max-w-full block"  // Adding block class here
        />
      </div>
      
      {/* Form */}
      <div className="flex justify-center w-full px-4 py-8">
        <form className="max-w-lg w-full">
          <div className="mb-4">
            <label htmlFor="name" className="block mb-2 font-bold">Name:</label>
            <input
              type="text"
              id="name"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="age" className="block mb-2 font-bold">Age:</label>
            <input
              type="number"
              id="age"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="mb-4">
            <label htmlFor="image" className="block mb-2 font-bold">Upload Image:</label>
          </div>
          <button
            type="submit"
            className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
          >
            Submit
          </button>
        </form>
      </div>
    </main>
  );
}
