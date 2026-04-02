import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "⬡ agentic",
  description: "Local-first, open-source Multi-Agent AI Runtime — powered by Ollama. Runs entirely on your machine.",
  icons: {
    icon: "/logo.svg",
    shortcut: "/logo.svg",
    apple: "/logo.svg",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
