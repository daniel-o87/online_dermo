import React from 'react';
import Link from 'next/link';

const Header: React.FC = () => {
  return (
    <header className="bg-gray-800 text-white p-4">
      <nav className="container mx-auto flex justify-between items-center">
        {/* Left-aligned logo */}
        <Link href="/" className="text-xl font-bold">
          Your Logo
        </Link>
        {/* Right-aligned navigation items */}
        <div className="flex ml-auto space-x-6 justify-end"> {/* Add justify-end */}
          <Link href="/" className="hover:text-gray-300 px-3 py-2">
            Home
          </Link>
          <Link href="/about" className="hover:text-gray-300 px-3 py-2">
            About
          </Link>
          <Link href="/faq" className="hover:text-gray-300 px-3 py-2">
            FAQ
          </Link>
          <Link href="/contact" className="hover:text-gray-300 px-3 py-2">
            Contact
          </Link>
        </div>
      </nav>
    </header>
  );
};

export default Header;
