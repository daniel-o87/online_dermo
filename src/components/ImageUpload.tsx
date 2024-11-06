import React, { ChangeEvent, useState } from 'react';

const ImageUpload: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const handleImageChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setSelectedImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        className="hidden"
        id="image"
      />
      <label
        htmlFor="image"
        className="inline-block bg-blue-500 text-white px-4 py-2 rounded-md cursor-pointer hover:bg-blue-600"
      >
        Choose Image
      </label>
      {selectedImage && (
        <div className="mt-4">
          <img src={selectedImage} alt="Selected" className="max-w-full h-auto rounded-md" />
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
