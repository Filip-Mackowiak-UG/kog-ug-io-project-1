"use client";

import React from "react";
import { ThemeToggle } from "./theme-toggle";
import Link from "next/link";
import { usePathname } from "next/navigation";

const pageTitles: { [key: string]: string } = {
  "/": "Peek a card 🫣",
  "/about": "About Us",
  "/contact": "Contact Us",
};

export default function Header() {
  const pathName = usePathname();

  const currPageTitle: string = pageTitles[pathName];

  return (
    <header className="grid grid-cols-3 items-center p-4 dark:bg-neutral-700 bg-neutral-300 border-b border-neutral-500 justify-between">
      <div className="flex">
        <Link href={"/"}>
          <h1 className="text-3xl font-bold">Keep On Gambling</h1>
        </Link>
      </div>
      <div className="text-center text-xl font-bold">{currPageTitle}</div>
      <div className="flex gap-8 items-center justify-end">
        <ul className="flex gap-4">
          {/* <li>zakładka 1</li>
          <li>zakładka 2</li>
          <li>zakładka 3</li> */}
        </ul>
        <ThemeToggle />
      </div>
    </header>
  );
}
