import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Agentic",
  description: "Local-first multi-agent AI runtime",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
