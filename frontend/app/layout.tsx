import './globals.css'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Melanoma Detection',
  description: 'Upload DICOM images for melanoma analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
