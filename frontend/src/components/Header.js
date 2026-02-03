import React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "./ThemeProvider";

const Header = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10 flex items-center justify-between px-6">
      <div>
        <h2 className="text-lg font-semibold text-foreground" data-testid="header-title">AI Campaign Dashboard</h2>
        <p className="text-xs text-muted-foreground">Orchestrate intelligent marketing campaigns</p>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" data-testid="status-indicator"></div>
          <span className="text-xs font-medium text-green-500">Active</span>
        </div>
        
        <button
          onClick={toggleTheme}
          data-testid="theme-toggle-button"
          className="p-2 rounded-lg bg-accent hover:bg-accent/80 transition-colors"
          aria-label="Toggle theme"
        >
          {theme === "dark" ? (
            <Sun size={18} className="text-accent-foreground" />
          ) : (
            <Moon size={18} className="text-accent-foreground" />
          )}
        </button>
      </div>
    </header>
  );
};

export default Header;
