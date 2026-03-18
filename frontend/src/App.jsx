import React, { useState } from 'react';
import axios from 'axios';
import {
  ShieldAlert, CheckCircle, Activity, User, TrendingUp, AlertTriangle,
  CalendarDays, CreditCard, Monitor, Lock, ShieldCheck, HelpCircle, Tv, Film
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, Cell, ReferenceLine
} from 'recharts';

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

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isPositive = data.value > 0;
    return (
      <div className="bg-slate-800 border border-slate-700 shadow-xl rounded-lg p-3">
        <p className="text-slate-300 text-xs font-semibold uppercase tracking-wider mb-1">{data.name}</p>
        <p className={`text-lg font-bold ${isPositive ? 'text-red-400' : 'text-emerald-400'}`}>
          {isPositive ? '+' : ''}{data.value.toFixed(3)}
        </p>
        <p className="text-slate-500 text-xs mt-1">
          {isPositive ? 'Increases churn risk' : 'Reduces churn risk'}
        </p>
      </div>
    );
  }
  return null;
};

function App() {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = "http://localhost:8000";

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    let newValue = type === 'number' ? parseFloat(value) : value;

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
      const predRes = await axios.post(`${API_URL}/predict`, formData);
      setPrediction(predRes.data);

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

    const shapData = explanation.feature_names.map((name, i) => {
      // Clean up feature names to make them highly readable
      let cleanName = name.replace(/_/g, ' ')
        .replace('Internet Service Fiber optic', 'Fiber Optic')
        .replace('Contract Month-to-month', 'Month-to-month Contract');

      // Capitalize first letter of each word
      cleanName = cleanName.replace(/\b\w/g, l => l.toUpperCase());

      return {
        name: cleanName,
        value: explanation.shap_values[i]
      };
    });

    // Sort by absolute importance to show the top 10
    shapData.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const topData = shapData.slice(0, 8); // Top 8 looks cleaner on the chart

    return (
      <div className="h-72 w-full mt-4">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={topData}
            margin={{ top: 10, right: 30, left: 140, bottom: 5 }}
            barSize={18}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} opacity={0.5} />
            <XAxis type="number" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={{ stroke: '#475569' }} tickLine={false} />
            <YAxis
              dataKey="name"
              type="category"
              stroke="#cbd5e1"
              width={130}
              tick={{ fill: '#e2e8f0', fontSize: 12, fontWeight: 500 }}
              axisLine={{ stroke: '#475569' }}
              tickLine={false}
            />
            <RechartsTooltip cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }} content={<CustomTooltip />} />
            <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={2} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {topData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#f43f5e' : '#10b981'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#0b1120] text-slate-50 p-4 md:p-8 font-sans selection:bg-blue-500/30">
      <div className="max-w-[1400px] mx-auto space-y-8">

        {/* Header */}
        <header className="flex flex-col md:flex-row items-start md:items-center justify-between pb-6 border-b border-slate-800">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-blue-500/10 rounded-xl border border-blue-500/20 shadow-lg shadow-blue-500/10">
              <Activity className="w-7 h-7 text-blue-400" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-white">
                Customer Retention Intelligence
              </h1>
              <p className="text-sm text-slate-400 font-medium tracking-wide mt-1">Predictive Churn Analysis Platform</p>
            </div>
          </div>

          <div className="mt-4 md:mt-0 flex items-center gap-3 bg-slate-800/50 px-4 py-2 rounded-full border border-slate-700/50">
            <div className={`w-2 h-2 rounded-full ${error ? 'bg-red-500' : 'bg-emerald-500 animate-pulse'}`}></div>
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-widest">System Model Online</span>
          </div>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">

          {/* Form Column */}
          <div className="xl:col-span-4 space-y-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-6 relative overflow-hidden shadow-2xl backdrop-blur-sm">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500"></div>

              <div className="flex items-center gap-3 mb-8 pb-4 border-b border-slate-800/80">
                <div className="p-2 bg-indigo-500/10 rounded-lg">
                  <User className="w-5 h-5 text-indigo-400" />
                </div>
                <h2 className="text-xl font-semibold text-slate-100">Profile Parameters</h2>
              </div>

              <form onSubmit={handlePredict} className="space-y-5">

                {/* Core Metrics */}
                <div className="space-y-4">
                  <div className="group">
                    <label className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                      <CalendarDays className="w-4 h-4 text-slate-500 group-hover:text-blue-400 transition-colors" />
                      Tenure (Months)
                    </label>
                    <input
                      type="number"
                      name="Tenure Months"
                      value={formData["Tenure Months"]}
                      onChange={handleChange}
                      className="w-full bg-[#0f172a] border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all font-medium"
                    />
                  </div>

                  <div className="group">
                    <label className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                      <CreditCard className="w-4 h-4 text-slate-500 group-hover:text-blue-400 transition-colors" />
                      Monthly Charges ($)
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      name="Monthly Charges"
                      value={formData["Monthly Charges"]}
                      onChange={handleChange}
                      className="w-full bg-[#0f172a] border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all font-medium"
                    />
                  </div>
                </div>

                {/* Dropdowns */}
                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div className="group">
                    <label className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                      <ShieldCheck className="w-4 h-4 text-slate-500 group-hover:text-indigo-400 transition-colors" />
                      Contract
                    </label>
                    <select
                      name="Contract"
                      value={formData.Contract}
                      onChange={handleChange}
                      className="w-full bg-[#0f172a] border border-slate-700 rounded-xl px-3 py-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all cursor-pointer"
                    >
                      <option>Month-to-month</option>
                      <option>One year</option>
                      <option>Two year</option>
                    </select>
                  </div>
                  <div className="group">
                    <label className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                      <Monitor className="w-4 h-4 text-slate-500 group-hover:text-indigo-400 transition-colors" />
                      Internet
                    </label>
                    <select
                      name="Internet Service"
                      value={formData["Internet Service"]}
                      onChange={handleChange}
                      className="w-full bg-[#0f172a] border border-slate-700 rounded-xl px-3 py-3 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all cursor-pointer"
                    >
                      <option>Fiber optic</option>
                      <option>DSL</option>
                      <option>No</option>
                    </select>
                  </div>
                </div>

                {/* Toggles */}
                <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 mt-4 space-y-4">
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Active Services</p>

                  <label className="flex items-center justify-between cursor-pointer group">
                    <div className="flex items-center gap-3">
                      <div className={`p-1.5 rounded-lg ${formData["Online Security"] === "Yes" ? 'bg-blue-500/20 text-blue-400' : 'bg-slate-800 text-slate-500'}`}>
                        <Lock className="w-4 h-4" />
                      </div>
                      <span className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">Online Security</span>
                    </div>
                    <div className={`w-10 h-5 rounded-full p-1 transition-colors duration-200 ease-in-out ${formData["Online Security"] === "Yes" ? 'bg-blue-500' : 'bg-slate-700'}`}>
                      <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform duration-200 ${formData["Online Security"] === "Yes" ? 'translate-x-5' : 'translate-x-0'}`}></div>
                    </div>
                    {/* Hidden input to handle state */}
                    <input type="checkbox" className="hidden" onChange={(e) => setFormData({ ...formData, "Online Security": formData["Online Security"] === "Yes" ? "No" : "Yes" })} />
                  </label>

                  <label className="flex items-center justify-between cursor-pointer group">
                    <div className="flex items-center gap-3">
                      <div className={`p-1.5 rounded-lg ${formData["Tech Support"] === "Yes" ? 'bg-purple-500/20 text-purple-400' : 'bg-slate-800 text-slate-500'}`}>
                        <HelpCircle className="w-4 h-4" />
                      </div>
                      <span className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">Tech Support</span>
                    </div>
                    <div className={`w-10 h-5 rounded-full p-1 transition-colors duration-200 ease-in-out ${formData["Tech Support"] === "Yes" ? 'bg-purple-500' : 'bg-slate-700'}`}>
                      <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform duration-200 ${formData["Tech Support"] === "Yes" ? 'translate-x-5' : 'translate-x-0'}`}></div>
                    </div>
                    <input type="checkbox" className="hidden" onChange={(e) => setFormData({ ...formData, "Tech Support": formData["Tech Support"] === "Yes" ? "No" : "Yes" })} />
                  </label>

                  <label className="flex items-center justify-between cursor-pointer group">
                    <div className="flex items-center gap-3">
                      <div className={`p-1.5 rounded-lg ${formData["Streaming TV"] === "Yes" ? 'bg-pink-500/20 text-pink-400' : 'bg-slate-800 text-slate-500'}`}>
                        <Tv className="w-4 h-4" />
                      </div>
                      <span className="text-sm font-medium text-slate-300 group-hover:text-white transition-colors">Streaming TV</span>
                    </div>
                    <div className={`w-10 h-5 rounded-full p-1 transition-colors duration-200 ease-in-out ${formData["Streaming TV"] === "Yes" ? 'bg-pink-500' : 'bg-slate-700'}`}>
                      <div className={`w-3 h-3 bg-white rounded-full shadow-md transform transition-transform duration-200 ${formData["Streaming TV"] === "Yes" ? 'translate-x-5' : 'translate-x-0'}`}></div>
                    </div>
                    <input type="checkbox" className="hidden" onChange={(e) => setFormData({ ...formData, "Streaming TV": formData["Streaming TV"] === "Yes" ? "No" : "Yes" })} />
                  </label>
                </div>

                <div className="pt-6">
                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-slate-100 hover:bg-white text-slate-900 font-bold py-3.5 px-4 rounded-xl shadow-lg shadow-white/10 transition-all transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <div className="w-5 h-5 border-2 border-slate-900/30 border-t-slate-900 rounded-full animate-spin" />
                    ) : (
                      <>
                        <Activity className="w-5 h-5" />
                        Run AI Inference
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Results Column */}
          <div className="xl:col-span-8 flex flex-col space-y-6">

            {error && (
              <div className="bg-red-500/10 border border-red-500/30 text-red-200 px-5 py-4 rounded-2xl flex items-start gap-3 shadow-lg shadow-red-500/5">
                <AlertTriangle className="w-6 h-6 flex-shrink-0 text-red-400 mt-0.5" />
                <div>
                  <h4 className="font-semibold text-red-300">Analysis Failed</h4>
                  <p className="text-sm mt-1 text-red-200/80">{error}</p>
                </div>
              </div>
            )}

            {!prediction && !loading && !error && (
              <div className="bg-slate-900/40 border border-slate-800 rounded-3xl flex flex-col items-center justify-center text-center flex-1 min-h-[500px] p-12 shadow-inner">
                <div className="w-20 h-20 bg-slate-800 rounded-2xl flex items-center justify-center mb-6 shadow-xl border border-slate-700/50">
                  <ShieldCheck className="w-10 h-10 text-slate-500" />
                </div>
                <h3 className="text-xl font-semibold text-slate-200 mb-2">Awaiting Parameters</h3>
                <p className="text-slate-400 max-w-sm leading-relaxed">
                  Configure the customer profile on the left and execute the AI inference model to generate retention probability and SHAP value explanations.
                </p>
              </div>
            )}

            {prediction && (
              <div className="space-y-6 flex-1 flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-700">

                {/* Score Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 relative overflow-hidden shadow-xl backdrop-blur-sm group">
                    <div className={`absolute -top-24 -right-24 w-48 h-48 rounded-full blur-3xl opacity-20 transition-colors duration-1000 ${prediction.churn_prob > 0.5 ? 'bg-red-500' : 'bg-emerald-500'}`}></div>
                    <div className="flex items-center justify-between mb-4">
                      <p className="text-sm font-semibold text-slate-400 uppercase tracking-widest">Calculated Risk Score</p>
                      <Activity className={`w-5 h-5 ${prediction.churn_prob > 0.5 ? 'text-red-400' : 'text-emerald-400'}`} />
                    </div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-6xl font-black tracking-tighter text-white">
                        {(prediction.churn_prob * 100).toFixed(1)}<span className="text-3xl text-slate-500">%</span>
                      </span>
                    </div>
                    <div className="mt-4 w-full bg-slate-800 rounded-full h-2 overflow-hidden border border-slate-700/50">
                      <div
                        className={`h-full rounded-full transition-all duration-1000 ${prediction.churn_prob > 0.5 ? 'bg-gradient-to-r from-orange-500 to-red-500' : 'bg-gradient-to-r from-emerald-400 to-teal-500'}`}
                        style={{ width: `${prediction.churn_prob * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className={`bg-slate-900/50 border rounded-3xl p-8 relative overflow-hidden shadow-xl backdrop-blur-sm transition-colors duration-1000 ${prediction.churn_prediction === 1 ? 'border-red-500/20' : 'border-emerald-500/20'}`}>
                    <div className={`absolute top-0 left-0 w-2 h-full ${prediction.churn_prediction === 1 ? 'bg-red-500' : 'bg-emerald-500'}`}></div>

                    <div className="flex flex-col h-full pl-4 justify-between">
                      <div>
                        <p className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1">Recommendation</p>
                        <h3 className={`text-3xl font-bold tracking-tight mt-2 ${prediction.churn_prediction === 1 ? 'text-red-400' : 'text-emerald-400'}`}>
                          {prediction.churn_prediction === 1 ? 'High Flight Risk' : 'Stable Customer'}
                        </h3>
                      </div>

                      <div className="mt-6 flex items-center gap-3">
                        {prediction.churn_prediction === 1 ? (
                          <div className="bg-red-500/10 text-red-300 px-4 py-2.5 rounded-xl font-medium text-sm border border-red-500/20 flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4" /> Trigger Retention Protocol
                          </div>
                        ) : (
                          <div className="bg-emerald-500/10 text-emerald-300 px-4 py-2.5 rounded-xl font-medium text-sm border border-emerald-500/20 flex items-center gap-2">
                            <CheckCircle className="w-4 h-4" /> No Action Required
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* SHAP Explanation */}
                {explanation && (
                  <div className="bg-slate-900/50 border border-slate-800 rounded-3xl p-8 flex-1 flex flex-col shadow-xl backdrop-blur-sm relative">
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6 pb-6 border-b border-slate-800/80">
                      <div>
                        <h3 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                          <TrendingUp className="w-5 h-5 text-blue-400" />
                          Feature Impact Analysis
                        </h3>
                        <p className="text-sm text-slate-400 mt-1">SHAP-based interpretability showing what drives this specific prediction</p>
                      </div>

                      <div className="flex items-center gap-6 bg-slate-800/50 px-5 py-2.5 rounded-2xl border border-slate-700/50 self-start">
                        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 tracking-wide">
                          <div className="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]"></div>
                          Increases Risk
                        </div>
                        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 tracking-wide">
                          <div className="w-3 h-3 rounded-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]"></div>
                          Reduces Risk
                        </div>
                      </div>
                    </div>

                    <div className="flex-1 min-h-[300px]">
                      {renderShapChart()}
                    </div>
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
