"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import React from "react";
import Markdown from "react-markdown";

type TabType = "Summary" | "Youtube" | "Reddit" | "Articles" | "CoinMarketCap";
type YoutubeData = Record<string, unknown>;
type RedditData = Record<string, unknown>;
type ArticlesData = Record<string, unknown>;
type SummaryData = Record<string, unknown>;
type CoinMarketCapData = Record<string, unknown>;

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("Summary");
  const [coinInput, setCoinInput] = useState("");
  const [currentCoin, setCurrentCoin] = useState("");
  const [sentimentScore, setSentimentScore] = useState<number | null>(null);
  const [loadedTabs, setLoadedTabs] = useState<Record<string, boolean>>({});

  const [loadingStates, setLoadingStates] = useState({
    Summary: false,
    Youtube: false,
    Reddit: false,
    Articles: false,
    CoinMarketCap: false,
  });

  const [youtubeData, setYoutubeData] = useState<YoutubeData | null>(null);
  const [redditData, setRedditData] = useState<RedditData | null>(null);
  const [articlesData, setArticlesData] = useState<ArticlesData | null>(null);
  const [summaryData, setSummaryData] = useState<SummaryData | null>(null);
  const [coinMarketData, setCoinMarketData] =
    useState<CoinMarketCapData | null>(null);

  const [errors, setErrors] = useState({
    Summary: null as string | null,
    Youtube: null as string | null,
    Reddit: null as string | null,
    Articles: null as string | null,
    CoinMarketCap: null as string | null,
  });

  const [redditSentiment, setRedditSentiment] = useState<number | null>(null);

  const fetchDataForTab = async (tab: TabType, coin: string) => {
    if (loadedTabs[tab]) return;

    setLoadingStates((prev) => ({ ...prev, [tab]: true }));
    setErrors((prev) => ({ ...prev, [tab]: null }));

    try {
      let endpoint = "";

      switch (tab) {
        case "Youtube":
          endpoint = "http://127.0.0.1:8000/v1/scrapeYoutube";
          break;
        case "Reddit":
          endpoint = "http://127.0.0.1:8000/v1/scrapeReddit";
          break;
        case "Articles":
          endpoint = "http://127.0.0.1:8000/v1/scrapeArticles";
          break;
        case "Summary":
          endpoint = "http://127.0.0.1:8000/v1/analyzeCoin";
          break;
        default:
          setCoinMarketData({
            message: `${tab} data fetch not implemented yet.`,
          });
          setLoadingStates((prev) => ({ ...prev, [tab]: false }));
          return;
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ coin }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch data: ${response.status}`);
      }

      const result = await response.json();

      if (result.sentiment !== undefined && tab === "Reddit") {
        setRedditSentiment(result.sentiment);
        if (activeTab === "Reddit") {
          setSentimentScore(result.sentiment);
        }
      }

      switch (tab) {
        case "Youtube":
          setYoutubeData(result);
          break;
        case "Reddit":
          setRedditData(result);
          break;
        case "Articles":
          setArticlesData(result);
          break;
        case "Summary":
          setSummaryData({ analysis: result.analysis });
          break;
        case "CoinMarketCap":
          setCoinMarketData(result);
          break;
      }

      setLoadedTabs((prev) => ({ ...prev, [tab]: true }));
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An unknown error occurred";
      setErrors((prev) => ({ ...prev, [tab]: errorMessage }));
    } finally {
      setLoadingStates((prev) => ({ ...prev, [tab]: false }));
    }
  };

  useEffect(() => {
    if (activeTab === "Reddit") {
      setSentimentScore(redditSentiment);
    } else {
      setSentimentScore(null);
    }
  }, [activeTab, redditSentiment]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!coinInput.trim()) return;

    const newCoin = coinInput.trim();

    if (newCoin !== currentCoin) {
      setLoadedTabs({});
      setCurrentCoin(newCoin);
    }

    fetchDataForTab(activeTab, newCoin);
  };

  const SentimentIndicator = ({ score }: { score: number | null }) => {
    if (score === null) return null;

    const getColorClass = () => {
      if (score > 0)
        return "bg-green-900/30 text-green-400 border-green-700/50";
      if (score < 0) return "bg-red-900/30 text-red-400 border-red-700/50";
      return "bg-gray-800 text-gray-400 border-gray-700/50";
    };

    const getLabel = () => {
      if (score > 0) return "Positive";
      if (score < 0) return "Negative";
      return "Neutral";
    };

    const getIcon = () => {
      if (score > 0) {
        return (
          <svg
            className="w-5 h-5 mr-2"
            fill="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M12 2C6.486 2 2 6.486 2 12s4.486 10 10 10 10-4.486 10-10S17.514 2 12 2zm0 18c-4.411 0-8-3.589-8-8s3.589-8 8-8 8 3.589 8 8-3.589 8-8 8zm4-6h-8a1 1 0 0 0 0 2h8a1 1 0 0 0 0-2zm-4-4a2 2 0 1 0-.001-4.001A2 2 0 0 0 12 10zm-4 0a2 2 0 1 0-.001-4.001A2 2 0 0 0 8 10zm8 0a2 2 0 1 0-.001-4.001A2 2 0 0 0 16 10z" />
          </svg>
        );
      }
      if (score < 0) {
        return (
          <svg
            className="w-5 h-5 mr-2"
            fill="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path d="M12 2C6.486 2 2 6.486 2 12s4.486 10 10 10 10-4.486 10-10S17.514 2 12 2zm0 18c-4.411 0-8-3.589-8-8s3.589-8 8-8 8 3.589 8 8-3.589 8-8 8zm4-9a2 2 0 1 0-.001 4.001A2 2 0 0 0 16 11zm-4 0a2 2 0 1 0-.001 4.001A2 2 0 0 0 12 11zm-4 0a2 2 0 1 0-.001 4.001A2 2 0 0 0 8 11zm4 6h8a1 1 0 0 0 0-2h-8a1 1 0 0 0 0 2z" />
          </svg>
        );
      }
      return (
        <svg
          className="w-5 h-5 mr-2"
          fill="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path d="M12 2C6.486 2 2 6.486 2 12s4.486 10 10 10 10-4.486 10-10S17.514 2 12 2zm0 18c-4.411 0-8-3.589-8-8s3.589-8 8-8 8 3.589 8 8-3.589 8-8 8zm-5-8h10a1 1 0 0 0 0-2H7a1 1 0 0 0 0 2z" />
        </svg>
      );
    };

    const absoluteScore = Math.abs(score);

    return (
      <div
        className={`flex items-center justify-between p-4 mb-4 border rounded-lg ${getColorClass()}`}
      >
        <div className="flex items-center">
          {getIcon()}
          <span className="font-medium">{getLabel()} Sentiment</span>
        </div>
        <div className="flex items-center">
          <div className="font-bold text-lg">
            {score > 0 ? "+" : ""}
            {score}
          </div>
          <div className="ml-3 w-24 bg-gray-700/50 h-2 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${score > 0 ? "bg-green-500" : score < 0 ? "bg-red-500" : "bg-gray-500"}`}
              style={{ width: `${Math.min(absoluteScore * 10, 100)}%` }}
            ></div>
          </div>
        </div>
      </div>
    );
  };

  const YoutubeContent = () => {
    const data = youtubeData;
    const isLoading = loadingStates.Youtube;
    const error = errors.Youtube;

    if (isLoading) {
      return <LoadingDisplay coin={currentCoin} />;
    }

    if (error) {
      return <ErrorDisplay error={error} />;
    }

    if (!data || Object.keys(data).length === 0) {
      return <EmptyDisplay type="YouTube videos" />;
    }

    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="grid grid-cols-1 md:grid-cols-2 gap-6"
      >
        {Object.values(data).map((item: any, index) => (
          <motion.div
            key={index}
            whileHover={{ scale: 1.02 }}
            className="bg-gray-800 rounded-lg overflow-hidden shadow-md border border-gray-700 hover:shadow-lg transition-all duration-300"
          >
            <div className="p-5">
              <h3 className="text-xl font-bold text-gray-100 mb-2 line-clamp-2">
                {item.title}
              </h3>

              <div className="flex items-center gap-2 mb-4">
                <div className="bg-red-600 rounded-full h-6 w-6 flex items-center justify-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="white"
                    className="w-3 h-3"
                  >
                    <path d="M15 11.063V7.6c0-.3-.125-.55-.375-.75-.25-.2-.55-.3-.9-.3H6.5l3.15 3.15-3.025 3.025a.651.651 0 0 0-.2.475c0 .183.067.35.2.5.15.133.317.2.5.2.2 0 .367-.067.5-.2L11 10.325l3.375 3.375c.133.133.3.2.5.2a.69.69 0 0 0 .5-.2.69.69 0 0 0 .2-.5c0-.2-.067-.367-.2-.5L12.35 9.675 15 7.025v4.038z"></path>
                  </svg>
                </div>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 text-sm font-medium flex items-center transition-colors"
                >
                  Watch on YouTube
                  <svg
                    className="w-4 h-4 ml-1"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                    ></path>
                  </svg>
                </a>
              </div>

              {item.Description && (
                <p className="text-gray-400 text-sm line-clamp-3">
                  {item.Description}
                </p>
              )}

              {item["Average Retention"] !== -1 && (
                <div className="mt-3 px-3 py-2 bg-gray-700/50 rounded-md">
                  <p className="text-xs text-gray-300">
                    Average Retention: {item["Average Retention"]}%
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </motion.div>
    );
  };

  const RedditContent = () => {
    const data = redditData;
    const isLoading = loadingStates.Reddit;
    const error = errors.Reddit;

    if (isLoading) {
      return <LoadingDisplay coin={currentCoin} />;
    }

    if (error) {
      return <ErrorDisplay error={error} />;
    }

    if (!data || Object.keys(data).length === 0) {
      return <EmptyDisplay type="Reddit posts" />;
    }

    const validData = Object.values(data).filter((item) => item && item.title);

    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="space-y-6"
      >
        {validData.map((item: any, index) => (
          <motion.div
            key={index}
            whileHover={{ scale: 1.01 }}
            className="bg-gray-800 rounded-lg overflow-hidden shadow-md border border-gray-700 hover:shadow-lg transition-all duration-300"
          >
            <div className="p-5">
              <div className="flex justify-between items-start mb-3">
                <h3 className="text-xl font-bold text-gray-100">
                  {item.title}
                </h3>
                <div className="flex items-center gap-2">
                  <div className="flex items-center">
                    <svg
                      className="w-4 h-4 text-orange-500"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path d="M12 2L8.66 9.34l-8.33 1.21 6.04 5.89-1.42 8.3L12 20.48l7.05 4.26-1.42-8.3 6.04-5.89-8.33-1.21z"></path>
                    </svg>
                    <span className="ml-1 text-sm font-medium text-gray-300">
                      {(item.upvote_ratio * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 mb-4">
                <div className="bg-orange-600 rounded-full h-6 w-6 flex items-center justify-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="white"
                    className="w-4 h-4"
                  >
                    <path d="M12 22c5.5 0 10-4.5 10-10S17.5 2 12 2 2 6.5 2 12s4.5 10 10 10zm1.2-15.5c1 0 1.8.8 1.8 1.8s-.8 1.8-1.8 1.8-1.8-.8-1.8-1.8.8-1.8 1.8-1.8zm2.8 9.3c-.4 1.3-1.5 2.2-2.8 2.2-1.4 0-2.6-.9-3-2.2-.1-.3.1-.6.4-.7.3-.1.6.1.7.4.3.8 1 1.4 1.9 1.4s1.6-.6 1.9-1.4c.1-.3.4-.5.7-.4.3.1.5.4.2.7z" />
                  </svg>
                </div>
                <a
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 text-sm font-medium flex items-center transition-colors"
                >
                  View on Reddit
                  <svg
                    className="w-4 h-4 ml-1"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                    ></path>
                  </svg>
                </a>
              </div>

              <div className="mt-4 prose max-w-none prose-invert prose-sm">
                {item.description ? (
                  <div className="prose prose-invert prose-sm max-w-none">
                    <Markdown>
                      {item.description.length > 200
                        ? `${item.description.slice(0, 200)}...`
                        : item.description}
                    </Markdown>

                    {item.description.length > 200 && (
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 hover:text-blue-300 text-sm font-medium transition-colors"
                      >
                        Read more
                      </a>
                    )}
                  </div>
                ) : (
                  <p>No description available.</p>
                )}
              </div>

              {item.comments && Object.keys(item.comments).length > 0 && (
                <div className="mt-6">
                  <h4 className="text-lg font-semibold text-gray-200 mb-3 flex items-center">
                    <svg
                      className="w-5 h-5 mr-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                      ></path>
                    </svg>
                    Top Comments
                  </h4>
                  <div className="space-y-3 mt-2">
                    {Object.values(item.comments)
                      .filter((comment) => comment && comment.text)
                      .map((comment: any, idx) => (
                        <motion.div
                          key={idx}
                          initial={{ opacity: 0, y: 5 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.2, delay: idx * 0.1 }}
                          className="p-3 bg-gray-700/50 rounded-lg"
                        >
                          <div className="prose prose-invert prose-sm max-w-none">
                            <Markdown>
                              {comment.text || "No comment text available."}
                            </Markdown>
                          </div>
                          <div className="flex gap-3 mt-2">
                            <span className="text-xs text-gray-400 flex items-center">
                              <svg
                                className="w-3 h-3 mr-1"
                                fill="currentColor"
                                viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg"
                              >
                                <path d="M12 4l1.4 1.4L7.8 11H20v2H7.8l5.6 5.6L12 20l-8-8 8-8z"></path>
                              </svg>
                              {comment.upvotes || 0}
                            </span>
                            <span
                              className={`text-xs px-2 py-0.5 rounded ${
                                comment.sentiment === "Positive"
                                  ? "bg-green-900/60 text-green-300"
                                  : comment.sentiment === "Negative"
                                    ? "bg-red-900/60 text-red-300"
                                    : "bg-gray-700 text-gray-400"
                              }`}
                            >
                              {comment.sentiment || "Neutral"}
                            </span>
                          </div>
                        </motion.div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </motion.div>
    );
  };

  const ArticlesContent = () => {
    const data = articlesData;
    const isLoading = loadingStates.Articles;
    const error = errors.Articles;

    if (isLoading) {
      return <LoadingDisplay coin={currentCoin} />;
    }

    if (error) {
      return <ErrorDisplay error={error} />;
    }

    if (!data || Object.keys(data).length === 0) {
      return <EmptyDisplay type="articles" />;
    }

    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="space-y-6"
      >
        {Object.values(data).map((item: any, index) => (
          <motion.div
            key={index}
            whileHover={{ scale: 1.01 }}
            className="bg-gray-800 rounded-lg overflow-hidden shadow-md border border-gray-700 hover:shadow-lg transition-all duration-300"
          >
            <div className="p-5">
              <h3 className="text-xl font-bold text-gray-100 mb-3">
                {item.title}
              </h3>

              <div className="flex items-center gap-2 mb-4">
                <div className="bg-blue-600 rounded-full h-6 w-6 flex items-center justify-center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="white"
                    className="w-4 h-4"
                  >
                    <path d="M12 2c5.5 0 10 4.5 10 10s-4.5 10-10 10S2 17.5 2 12 6.5 2 12 2zm-1.5 5v6.5H17v-2h-4.5V7h-2z" />
                  </svg>
                </div>
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 text-sm font-medium flex items-center transition-colors"
                >
                  Read full article
                  <svg
                    className="w-4 h-4 ml-1"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                    ></path>
                  </svg>
                </a>
              </div>

              {item.summary && (
                <div className="mt-4 p-4 bg-gray-700/50 rounded-lg border border-gray-700">
                  <h4 className="font-medium text-gray-200 mb-2 flex items-center">
                    <svg
                      className="w-4 h-4 mr-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      ></path>
                    </svg>
                    Summary
                  </h4>
                  <p className="text-gray-300 text-sm">{item.summary}</p>
                </div>
              )}

              {item.keywords && item.keywords.length > 0 && (
                <div className="mt-4">
                  <h4 className="font-medium text-gray-200 mb-2 flex items-center">
                    <svg
                      className="w-4 h-4 mr-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
                      ></path>
                    </svg>
                    Keywords
                  </h4>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {item.keywords.map((keyword: string, idx: number) => (
                      <span
                        key={idx}
                        className="text-xs bg-blue-900/40 text-blue-300 px-2 py-1 rounded-full border border-blue-800/50"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </motion.div>
    );
  };

  const SummaryContent = () => {
    const data = summaryData;
    const isLoading = loadingStates.Summary;
    const error = errors.Summary;

    if (isLoading) {
      return <LoadingDisplay coin={currentCoin} />;
    }

    if (error) {
      return <ErrorDisplay error={error} />;
    }

    if (!data || !data.analysis) {
      return (
        <div className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center">
          <svg
            className="w-16 h-16 mx-auto text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            ></path>
          </svg>
          <h3 className="mt-4 text-xl font-medium text-gray-300">
            Summary Analysis
          </h3>
          <p className="mt-2 text-gray-400">No summary data available.</p>
        </div>
      );
    }

    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="bg-gray-800 rounded-lg p-8 border border-gray-700"
      >
        <h3 className="text-xl font-medium text-gray-300 mb-4 flex items-center">
          <svg
            className="w-5 h-5 mr-2 text-blue-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M13 10V3L4 14h7v7l9-11h-7z"
            ></path>
          </svg>
          Summary Analysis
        </h3>
        <div className="prose prose-invert prose-lg max-w-none">
          <Markdown
            components={{
              h1: ({ ...props }) => (
                <h1
                  className="text-2xl font-bold text-blue-400 mt-6 mb-4"
                  {...props}
                />
              ),
              h2: ({ ...props }) => (
                <h2
                  className="text-xl font-bold text-blue-300 mt-5 mb-3"
                  {...props}
                />
              ),
              h3: ({ ...props }) => (
                <h3
                  className="text-lg font-semibold text-blue-200 mt-4 mb-2"
                  {...props}
                />
              ),
              h4: ({ ...props }) => (
                <h4
                  className="text-base font-semibold text-gray-200 mt-3 mb-2"
                  {...props}
                />
              ),
              p: ({ ...props }) => (
                <p className="my-3 text-gray-300 leading-relaxed" {...props} />
              ),
              ul: ({ ...props }) => (
                <ul className="list-disc pl-6 my-3 space-y-1" {...props} />
              ),
              ol: ({ ...props }) => (
                <ol className="list-decimal pl-6 my-3 space-y-1" {...props} />
              ),
              li: ({ ...props }) => (
                <li className="text-gray-300 my-1" {...props} />
              ),
              blockquote: ({ ...props }) => (
                <blockquote
                  className="border-l-4 border-blue-700 pl-4 italic text-gray-400 my-4"
                  {...props}
                />
              ),
              code: ({
                inline,
                ...props
              }: {
                inline?: boolean;
                children?: React.ReactNode;
              }) =>
                inline ? (
                  <code
                    className="bg-gray-700 px-1 py-0.5 rounded text-sm text-orange-300"
                    {...props}
                  />
                ) : (
                  <code
                    className="block bg-gray-700/50 p-4 rounded-md my-4 overflow-auto text-sm text-orange-300"
                    {...props}
                  />
                ),
            }}
          >
            {data.analysis}
          </Markdown>
        </div>
      </motion.div>
    );
  };

  const CoinMarketCapContent = () => {
    const isLoading = loadingStates.CoinMarketCap;
    const error = errors.CoinMarketCap;

    if (isLoading) {
      return <LoadingDisplay coin={currentCoin} />;
    }

    if (error) {
      return <ErrorDisplay error={error} />;
    }

    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="bg-gray-800 rounded-lg p-8 border border-gray-700 text-center"
      >
        <svg
          className="w-16 h-16 mx-auto text-gray-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          ></path>
        </svg>
        <h3 className="mt-4 text-xl font-medium text-gray-300">
          CoinMarketCap Integration
        </h3>
        <p className="mt-2 text-gray-400">
          Market data will be available soon!
        </p>
      </motion.div>
    );
  };

  const LoadingDisplay = ({ coin }: { coin: string }) => (
    <div className="flex flex-col items-center justify-center p-12">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
      <p className="text-gray-400">Analyzing {coin}...</p>
    </div>
  );

  const ErrorDisplay = ({ error }: { error: string }) => (
    <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 text-center">
      <svg
        className="w-12 h-12 mx-auto text-red-500 mb-2"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        ></path>
      </svg>
      <h3 className="text-lg font-medium text-red-300 mb-1">Error</h3>
      <p className="text-red-200">{error}</p>
    </div>
  );

  const EmptyDisplay = ({ type }: { type: string }) => (
    <div className="bg-gray-800/50 rounded-lg p-8 text-center border border-gray-700">
      <svg
        className="w-12 h-12 mx-auto text-gray-600 mb-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth="2"
          d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"
        ></path>
      </svg>
      <h3 className="text-lg font-medium text-gray-400">No {type} found</h3>
      <p className="text-gray-500 mt-1">
        Try searching for a different cryptocurrency or check back later.
      </p>
    </div>
  );

  const renderContent = () => {
    if (!currentCoin) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-center">
          <svg
            className="w-16 h-16 text-gray-700 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            ></path>
          </svg>
          <h3 className="text-xl font-medium text-gray-400 mb-2">
            Enter a cryptocurrency name
          </h3>
          <p className="text-gray-500 max-w-md">
            Type the name of any cryptocurrency (like Bitcoin, Ethereum, or
            Dogecoin) and click &quot;Analyze&quot; to see insights.
          </p>
        </div>
      );
    }

    switch (activeTab) {
      case "Youtube":
        return <YoutubeContent />;
      case "Reddit":
        return <RedditContent />;
      case "Articles":
        return <ArticlesContent />;
      case "Summary":
        return <SummaryContent />;
      case "CoinMarketCap":
        return <CoinMarketCapContent />;
      default:
        return <div>Select a tab to view data</div>;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-950 text-gray-200">
      <div className="max-w-7xl mx-auto p-4 sm:p-6">
        <header className="pt-6 pb-10">
          <div className="flex flex-col items-center mb-8">
            <div className="flex items-center gap-3 mb-2">
              <div className="bg-blue-500 h-8 w-8 rounded-lg flex items-center justify-center">
                <svg
                  className="w-5 h-5 text-white"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path d="M8.433 7.418c.155-.103.346-.196.567-.267v1.698a2.305 2.305 0 01-.567-.267C8.07 8.34 8 8.114 8 8c0-.114.07-.34.433-.582zM11 12.849v-1.698c.22.071.412.164.567.267.364.243.433.468.433.582 0 .114-.07.34-.433.582a2.305 2.305 0 01-.567.267z" />
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-13a1 1 0 10-2 0v.092a4.535 4.535 0 00-1.676.662C6.602 6.234 6 7.009 6 8c0 .99.602 1.765 1.324 2.246.48.32 1.054.545 1.676.662v1.941c-.391-.127-.68-.317-.843-.504a1 1 0 10-1.51 1.31c.562.649 1.413 1.076 2.353 1.253V15a1 1 0 102 0v-.092a4.535 4.535 0 001.676-.662C13.398 13.766 14 12.991 14 12c0-.99-.602-1.765-1.324-2.246A4.535 4.535 0 0011 9.092V7.151c.391.127.68.317.843.504a1 1 0 101.511-1.31c-.563-.649-1.413-1.076-2.354-1.253V5z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text tracking-tight">
                DeepCoin
              </h1>
            </div>
            <p className="text-gray-400">
              AI-Powered Cryptocurrency Sentiment Analysis
            </p>
          </div>

          <form
            onSubmit={handleSubmit}
            className="flex flex-col sm:flex-row gap-3 max-w-2xl mx-auto"
          >
            <div className="relative flex-grow">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg
                  className="w-5 h-5 text-gray-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                  ></path>
                </svg>
              </div>
              <input
                type="text"
                value={coinInput}
                onChange={(e) => setCoinInput(e.target.value)}
                placeholder="Enter cryptocurrency name (e.g. Bitcoin)"
                className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all text-gray-200"
                required
              />
            </div>
            <motion.button
              type="submit"
              disabled={loadingStates[activeTab]}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-500 text-white font-medium rounded-lg shadow hover:shadow-lg hover:from-blue-500 hover:to-blue-400 transition-all ${
                loadingStates[activeTab] ? "opacity-70 cursor-not-allowed" : ""
              }`}
            >
              {loadingStates[activeTab] ? (
                <div className="flex items-center justify-center">
                  <svg
                    className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Analyzing...
                </div>
              ) : (
                <div className="flex items-center">
                  <svg
                    className="w-5 h-5 mr-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    ></path>
                  </svg>
                  Analyze
                </div>
              )}
            </motion.button>
          </form>
        </header>

        <div className="mb-6">
          <nav className="flex overflow-x-auto scrollbar-hide">
            <div className="bg-gray-800/80 backdrop-blur rounded-lg p-1 flex space-x-1 mx-auto">
              {[
                "Summary",
                "Youtube",
                "Reddit",
                "Articles",
                "CoinMarketCap",
              ].map((tab) => (
                <motion.button
                  key={tab}
                  onClick={() => {
                    setActiveTab(tab as TabType);
                    if (currentCoin && !loadedTabs[tab as TabType]) {
                      fetchDataForTab(tab as TabType, currentCoin);
                    }
                  }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`px-4 py-2 rounded-md font-medium whitespace-nowrap text-sm transition-colors ${
                    activeTab === tab
                      ? "bg-blue-600 text-white shadow-md"
                      : "text-gray-400 hover:text-gray-200 hover:bg-gray-700/50"
                  }`}
                >
                  {tab}
                </motion.button>
              ))}
            </div>
          </nav>
        </div>

        <main className="bg-gray-800/70 backdrop-blur-sm rounded-xl shadow-xl overflow-hidden border border-gray-700/50">
          <div className="p-6">
            {currentCoin && (
              <div className="mb-6 pb-4 border-b border-gray-700/70">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="mr-3">
                      <div className="bg-blue-900/50 h-10 w-10 rounded-full flex items-center justify-center border border-blue-700/50">
                        <span className="text-lg font-bold text-blue-400">
                          {currentCoin.charAt(0).toUpperCase()}
                        </span>
                      </div>
                    </div>
                    <div>
                      <h2 className="text-xl font-bold text-gray-100">
                        {currentCoin}{" "}
                        <span className="text-gray-500 font-normal text-sm">
                          analysis
                        </span>
                      </h2>
                      <p className="text-sm text-gray-400">
                        {activeTab} insights
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {sentimentScore !== null && activeTab === "Reddit" && (
              <SentimentIndicator score={sentimentScore} />
            )}

            <div className="min-h-[400px]">{renderContent()}</div>
          </div>
        </main>

        <footer className="mt-12 text-center text-gray-500 text-sm py-6">
          <p>
            © {new Date().getFullYear()} DeepCoin • AI-Powered Cryptocurrency
            Analysis
          </p>
        </footer>
      </div>
    </div>
  );
}
