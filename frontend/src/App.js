import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./components/ThemeProvider";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
import CampaignRunner from "./pages/CampaignRunner";
import VectorMemory from "./pages/VectorMemory";
import SelfReflection from "./pages/SelfReflection";
import Analytics from "./pages/Analytics";
import CampaignHistory from "./pages/CampaignHistory";
import CampaignDetail from "./pages/CampaignDetail";

function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="flex min-h-screen bg-background">
          <Sidebar />
          <div className="flex-1 flex flex-col">
            <Header />
            <main className="flex-1 p-8 overflow-auto">
              <Routes>
                <Route path="/" element={<CampaignRunner />} />
                <Route path="/vector-memory" element={<VectorMemory />} />
                <Route path="/self-reflection" element={<SelfReflection />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/history" element={<CampaignHistory />} />
                <Route path="/campaign/:campaignId" element={<CampaignDetail />} />
              </Routes>
            </main>
          </div>
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
