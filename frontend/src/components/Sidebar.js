import React from "react";
import { Link, useLocation } from "react-router-dom";
import { 
  PlayCircle, 
  Database, 
  BarChart3, 
  History, 
  Brain 
} from "lucide-react";

const Sidebar = () => {
  const location = useLocation();
  
  const navItems = [
    { path: "/", icon: PlayCircle, label: "Campaign Runner", testId: "nav-campaign-runner" },
    { path: "/vector-memory", icon: Database, label: "Vector Memory", testId: "nav-vector-memory" },
    { path: "/self-reflection", icon: Brain, label: "Self Reflection", testId: "nav-self-reflection" },
    { path: "/analytics", icon: BarChart3, label: "Analytics", testId: "nav-analytics" },
    { path: "/history", icon: History, label: "Campaign History", testId: "nav-history" },
  ];

  return (
    <aside className="w-64 bg-card border-r border-border h-screen sticky top-0 flex flex-col">
      <div className="p-6 border-b border-border">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-cyan-400 bg-clip-text text-transparent" data-testid="sidebar-logo">
          UHPM
        </h1>
        <p className="text-xs text-muted-foreground mt-1">AI Campaign Orchestration</p>
      </div>
      
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          
          return (
            <Link
              key={item.path}
              to={item.path}
              data-testid={item.testId}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                isActive
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              }`}
            >
              <Icon size={20} />
              <span className="font-medium">{item.label}</span>
            </Link>
          );
        })}
      </nav>
      
      <div className="p-4 border-t border-border">
        <div className="flex items-center space-x-2 text-sm text-muted-foreground">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span>System Online</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
