import React, { useState } from 'react';
import axios from 'axios';
import { ShieldAlert, CheckCircle, Activity, User, TrendingUp, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

const INITIAL_FORM = {
  CustomerID: "CUST-001",
  Gender: "Male",
  Senior_Citizen: "No",
  Partner: "No",
  Dependents: "No",
  "Tenure Months": 12,
  "Phone Service": "Yes",
  "Multiple Lines": "No",
  "Internet Service": "Fiber optic",
  "Online Security": "No",
  "Online Backup": "No",
  "Device Protection": "No",
  "Tech Support": "No",
  "Streaming TV": "Yes",
  "Streaming Movies": "Yes",
  Contract: "Month-to-month",
  "Paperless Billing": "Yes",
  "Payment Method": "Electronic check",
  "Monthly Charges": 89.50,
  "Total Charges": 1074.00
};

function App() {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = "http://localhost:8000";

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    let newValue = value;

    // For numeric inputs
    if (type === 'number') {
      newValue = parseFloat(value);
    }

    setFormData((prev) => ({
      ...prev,
      [name]: newValue
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);
    setExplanation(null);

    try {
      // 1. Get Prediction
      const predRes = await axios.post(`${API_URL}/predict`, formData);
      setPrediction(predRes.data);

      // 2. Get Explanation (SHAP)
      const expRes = await axios.post(`${API_URL}/explain`, formData);
      setExplanation(expRes.data);

    } catch (err) {
      setError(err.response?.data?.detail || err.message || "An error occurred connecting to the API.");
    } finally {
      setLoading(false);
    }
  };

  const renderShapChart = () => {
    if (!explanation) return null;

    const shapData = explanation.feature_names.map((name, i) => ({
      name: name.replace(/_/g, ' '),
      value: explanation.shap_values[i]
    }));

    // Sort by absolute importance to show the top 10
    shapData.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const topData = shapData.slice(0, 10);

    return (
      <div className="h-80 w-full mt-6">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={topData}
            margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis type="number" stroke="#94a3b8" />
            <YAxis dataKey="name" type="category" stroke="#94a3b8" width={100} tick={{ fontSize: 12 }} />
            <RechartsTooltip
              contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
              itemStyle={{ color: '#f8fafc' }}
            />
            <ReferenceLine x={0} stroke="#94a3b8" />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {topData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-50 p-6 md:p-10 font-sans">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* Header */}
        <header className="flex items-center justify-between pb-6 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-500" />
            <h1 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
              CRIP Dashboard
            </h1>
          </div>
          <p className="text-slate-400 font-medium">Customer Retention Intelligence Platform</p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

          {/* Form Column */}
          <div className="lg:col-span-5 space-y-6">
            <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-blue-500 to-indigo-600"></div>

              <div className="flex items-center gap-2 mb-6">
                <User className="w-5 h-5 text-indigo-400" />
                <h2 className="text-xl font-semibold">Customer Profile</h2>
              </div>

              <form onSubmit={handlePredict} className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Tenure (Months)</label>
                    <input
                      type="number"
                      name="Tenure Months"
                      value={formData["Tenure Months"]}
                      onChange={handleChange}
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                    />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Monthly Charges</label>
                    <input
                      type="number"
                      step="0.01"
                      name="Monthly Charges"
                      value={formData["Monthly Charges"]}
                      onChange={handleChange}
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Contract Type</label>
                    <select
                      name="Contract"
                      value={formData.Contract}
                      onChange={handleChange}
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                    >
                      <option>Month-to-month</option>
                      <option>One year</option>
                      <option>Two year</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Internet Service</label>
                    <select
                      name="Internet Service"
                      value={formData["Internet Service"]}
                      onChange={handleChange}
                      className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
                    >
                      <option>Fiber optic</option>
                      <option>DSL</option>
                      <option>No</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1 flex items-center gap-2 pt-4">
                    <input type="checkbox" name="Online Security" checked={formData["Online Security"] === "Yes"} onChange={(e) => setFormData({ ...formData, "Online Security": e.target.checked ? "Yes" : "No" })} className="w-4 h-4 rounded bg-slate-800 border-slate-600 text-blue-500 focus:ring-blue-500" />
                    <label className="text-sm font-medium text-slate-300">Online Security</label>
                  </div>
                  <div className="space-y-1 flex items-center gap-2 pt-4">
                    <input type="checkbox" name="Tech Support" checked={formData["Tech Support"] === "Yes"} onChange={(e) => setFormData({ ...formData, "Tech Support": e.target.checked ? "Yes" : "No" })} className="w-4 h-4 rounded bg-slate-800 border-slate-600 text-blue-500 focus:ring-blue-500" />
                    <label className="text-sm font-medium text-slate-300">Tech Support</label>
                  </div>
                </div>

                <div className="pt-6">
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium py-3 px-4 rounded-xl shadow-lg shadow-blue-900/20 transition-all transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <>
                        <TrendingUp className="w-5 h-5" />
                        Analyze Churn Risk
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Results Column */}
          <div className="lg:col-span-7 space-y-6">

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-xl flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <p className="text-sm">{error}</p>
              </div>
            )}

            {!prediction && !loading && !error && (
              <div className="glass-panel p-12 rounded-2xl flex flex-col items-center justify-center text-center h-full border-dashed border-slate-600">
                <div className="w-16 h-16 bg-slate-800/50 rounded-full flex items-center justify-center mb-4">
                  <ShieldAlert className="w-8 h-8 text-slate-500" />
                </div>
                <h3 className="text-lg font-medium text-slate-300">No Prediction Yet</h3>
                <p className="text-sm text-slate-500 mt-2 max-w-sm">
                  Adjust the customer profile parameters and click Analyze to generate a churn risk assessment.
                </p>
              </div>
            )}

            {prediction && (
              <div className="space-y-6 fade-in">
                {/* Score Cards */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="glass-panel p-6 rounded-2xl relative overflow-hidden">
                    <div className={`absolute top-0 right-0 w-16 h-16 blur-2xl rounded-full -mr-8 -mt-8 ${prediction.churn_prob > 0.5 ? 'bg-red-500/30' : 'bg-emerald-500/30'}`}></div>
                    <p className="text-sm font-medium text-slate-400 mb-1">Churn Probability</p>
                    <div className="flex items-baseline gap-2">
                      <span className="text-4xl font-bold tracking-tight">
                        {(prediction.churn_prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className={`glass-panel p-6 rounded-2xl border ${prediction.churn_prediction === 1 ? 'border-red-500/30 bg-red-500/5' : 'border-emerald-500/30 bg-emerald-500/5'}`}>
                    <div className="flex flex-col justify-center h-full">
                      <p className="text-sm font-medium text-slate-400 mb-2">Risk Status</p>
                      <div className="flex items-center gap-2">
                        {prediction.churn_prediction === 1 ? (
                          <>
                            <AlertTriangle className="w-6 h-6 text-red-400" />
                            <span className="text-xl font-semibold text-red-400">High Risk (Churn)</span>
                          </>
                        ) : (
                          <>
                            <CheckCircle className="w-6 h-6 text-emerald-400" />
                            <span className="text-xl font-semibold text-emerald-400">Low Risk (Retain)</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* SHAP Explanation */}
                {explanation && (
                  <div className="glass-panel p-6 rounded-2xl h-[400px]">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold border-b border-transparent">Why this prediction?</h3>
                      <div className="flex items-center gap-4 text-xs font-medium text-slate-400">
                        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-red-500"></div> Increases risk</div>
                        <div className="flex items-center gap-1.5"><div className="w-3 h-3 rounded-full bg-emerald-500"></div> Decreases risk</div>
                      </div>
                    </div>
                    {renderShapChart()}
                  </div>
                )}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;
