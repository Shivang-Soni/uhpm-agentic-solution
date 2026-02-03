import React from "react";
import { Line, Doughnut, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from "chart.js";
import { TrendingUp, Clock, Target } from "lucide-react";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const Analytics = () => {
  const successRateData = {
    labels: ["PLAN", "RESEARCH", "PERSONA", "CONTENT", "EXPERIMENT", "PREVIEW", "PUBLISH", "ANALYTICS"],
    datasets: [
      {
        label: "Success Rate (%)",
        data: [95, 88, 92, 85, 78, 90, 87, 94],
        borderColor: "rgb(6, 182, 212)",
        backgroundColor: "rgba(6, 182, 212, 0.1)",
        tension: 0.4,
        fill: true,
      },
    ],
  };

  const retryCountData = {
    labels: ["PLAN", "RESEARCH", "PERSONA", "CONTENT", "EXPERIMENT", "PREVIEW", "PUBLISH", "ANALYTICS"],
    datasets: [
      {
        label: "Retry Count",
        data: [0, 2, 1, 3, 4, 1, 2, 0],
        backgroundColor: "rgba(59, 130, 246, 0.8)",
        borderColor: "rgba(59, 130, 246, 1)",
        borderWidth: 1,
      },
    ],
  };

  const executionDurationData = {
    labels: ["Week 1", "Week 2", "Week 3", "Week 4"],
    datasets: [
      {
        label: "Avg Duration (minutes)",
        data: [45, 38, 42, 35],
        borderColor: "rgb(16, 185, 129)",
        backgroundColor: "rgba(16, 185, 129, 0.1)",
        tension: 0.4,
        fill: true,
      },
    ],
  };

  const statusDistribution = {
    labels: ["Success", "Failed", "Running"],
    datasets: [
      {
        data: [75, 15, 10],
        backgroundColor: [
          "rgba(16, 185, 129, 0.8)",
          "rgba(239, 68, 68, 0.8)",
          "rgba(59, 130, 246, 0.8)",
        ],
        borderColor: [
          "rgb(16, 185, 129)",
          "rgb(239, 68, 68)",
          "rgb(59, 130, 246)",
        ],
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: "rgb(156, 163, 175)",
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: "rgb(156, 163, 175)",
        },
        grid: {
          color: "rgba(156, 163, 175, 0.1)",
        },
      },
      y: {
        ticks: {
          color: "rgb(156, 163, 175)",
        },
        grid: {
          color: "rgba(156, 163, 175, 0.1)",
        },
      },
    },
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: "bottom",
        labels: {
          color: "rgb(156, 163, 175)",
          padding: 20,
        },
      },
    },
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground" data-testid="analytics-title">Analytics</h1>
        <p className="text-muted-foreground mt-1">Campaign performance metrics and insights</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold text-foreground" data-testid="total-campaigns">156</div>
              <div className="text-sm text-muted-foreground mt-1">Total Campaigns</div>
            </div>
            <div className="p-3 bg-primary/10 rounded-lg">
              <Target size={24} className="text-primary" />
            </div>
          </div>
        </div>
        
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold text-foreground" data-testid="avg-success-rate">89%</div>
              <div className="text-sm text-muted-foreground mt-1">Avg Success Rate</div>
            </div>
            <div className="p-3 bg-green-500/10 rounded-lg">
              <TrendingUp size={24} className="text-green-500" />
            </div>
          </div>
        </div>
        
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl font-bold text-foreground" data-testid="avg-duration">40m</div>
              <div className="text-sm text-muted-foreground mt-1">Avg Duration</div>
            </div>
            <div className="p-3 bg-cyan-500/10 rounded-lg">
              <Clock size={24} className="text-cyan-500" />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-foreground mb-4">Action Success Rate</h2>
          <div className="h-64" data-testid="success-rate-chart">
            <Line data={successRateData} options={chartOptions} />
          </div>
        </div>

        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-foreground mb-4">Retry Count per Agent</h2>
          <div className="h-64" data-testid="retry-count-chart">
            <Bar data={retryCountData} options={chartOptions} />
          </div>
        </div>

        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-foreground mb-4">Execution Duration Trend</h2>
          <div className="h-64" data-testid="duration-chart">
            <Line data={executionDurationData} options={chartOptions} />
          </div>
        </div>

        <div className="bg-card rounded-lg border border-border p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-foreground mb-4">Campaign Status Distribution</h2>
          <div className="h-64" data-testid="status-distribution-chart">
            <Doughnut data={statusDistribution} options={doughnutOptions} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
