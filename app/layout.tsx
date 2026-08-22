import "./globals.css";

export const metadata = {
  title: "YUAKE",
  description: "A camera-driven interactive reality."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
