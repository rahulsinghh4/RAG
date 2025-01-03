import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import cx from "@/utils/cx";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  icons: {
    icon: "/favicon.ico",
  },
  title: "Physics RAG App",
  description: "Physics Scientific Literature Chat Bot",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth antialiased">
      <body
        className={cx(
          inter.className,
          "text-sm md:text-base bg-zinc-800 text-zinc-50",
        )}
      >
        {children}
      </body>
    </html>
  );
}
