from agents.market_data_agent import MarketDataAgent
from agents.tech_analysis_agent import TechnicalAnalysisAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from database.models import create_tables
import time
import asyncio

async def main():
    print("🚀 Starting Portfolio Bot MVP...")
    
    # # Create database tables
    # create_tables()
    # print("✅ Database initialized")
    
    # # Initialize market data agent with our 5 stocks
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    # market_agent = MarketDataAgent(symbols)
    # print(f"✅ Market Data Agent created for symbols: {symbols}")
    
    # # Test the agent
    # print("\n📈 Fetching current market data...")
    # result = market_agent.process()
    
    # if result["status"] == "success":
    #     print("\n🎉 SUCCESS! Current prices:")
    #     for symbol, (price, timestamp) in result["prices"].items():
    #         print(f"  {symbol}: ${price:.2f}")
    # else:
    #     print(f"\n❌ Status: {result['status']}")
    #     print("Prices from historic data:")
    #     for symbol, price_list in result["prices"].items():
    #         for price, timestamp in price_list:
    #             print(f"  {symbol}: ${price:.2f} at {timestamp.strftime('%Y-%m-%d')}")

    
    # print(f"\n🔄 Market open: {market_agent.is_market_open()}")
    # print(f"🕐 Last update: {market_agent.last_update}")

    # print("Performing technical analysis...")
    # tech_agent = TechnicalAnalysisAgent(symbols)
    # analysis_results = tech_agent.process()
    # for symbol, analysis in analysis_results.items():
    #     print(f"{symbol}: {analysis_results[symbol]}")

    news_agent = NewsSentimentAgent(symbols)
    results = await news_agent.process()
    print(results)



if __name__ == "__main__":
    asyncio.run(main())