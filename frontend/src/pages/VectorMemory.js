import React from "react";
import { Database, TrendingUp, Users } from "lucide-react";

const VectorMemory = () => {
  const memoryCards = [
    {
      title: "Similar Past Campaigns",
      icon: Database,
      data: [
        { text: "Product launch campaign with 45% conversion increase", distance: 0.12 },
        { text: "Seasonal promotion targeting millennials - 3.2x ROI", distance: 0.15 },
        { text: "Brand awareness campaign across social platforms", distance: 0.19 },
      ]
    },
    {
      title: "Successful Creatives",
      icon: TrendingUp,
      data: [
        { text: "Minimalist design with bold CTA - 67% click-through rate", distance: 0.18 },
        { text: "Video ad with product demo - 89% engagement", distance: 0.21 },
        { text: "Carousel format showcasing features - 54% conversion", distance: 0.23 },
      ]
    },
    {
      title: "Audience Insights",
      icon: Users,
      data: [
        { text: "Tech-savvy millennials, mobile-first behavior", distance: 0.24 },
        { text: "Price-sensitive audience, responds to discounts", distance: 0.27 },
        { text: "Early adopters, interested in innovation", distance: 0.29 },
      ]
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground" data-testid="vector-memory-title">Vector Memory</h1>
        <p className="text-muted-foreground mt-1">Retrieved knowledge from past campaigns and insights</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {memoryCards.map((card, idx) => {
          const Icon = card.icon;
          return (
            <div 
              key={idx} 
              className="bg-card rounded-lg border border-border p-6 shadow-sm hover:shadow-md transition-shadow"
              data-testid={`vector-memory-card-${idx}`}
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Icon size={20} className="text-primary" />
                </div>
                <h2 className="text-lg font-semibold text-foreground">{card.title}</h2>
              </div>

              <div className="space-y-3">
                {card.data.map((item, itemIdx) => (
                  <div 
                    key={itemIdx} 
                    className="p-3 bg-accent/50 rounded-lg border border-border hover:border-primary/50 transition-colors"
                    data-testid={`vector-item-${idx}-${itemIdx}`}
                  >
                    <p className="text-sm text-foreground mb-2">{item.text}</p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Distance Score</span>
                      <span className="text-xs font-mono text-primary">{item.distance.toFixed(3)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default VectorMemory;
