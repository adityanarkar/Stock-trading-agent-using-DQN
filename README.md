# Stock-trading-agent-using-DQN

I applied DQN on stock market data. I preprocessed data available on www.quandl.com, and used that to train our agent. I developed two agents, one with 2 actions - Buy and Sell, and the other one with 3 actions - Buy, Hold and Sell and tried to see which will be stable and perform better.

I followed research paper by Jonah Varon and Anthony Soroka, titled "Stock Trading with Reinforcement Learning". Their method was to use 2 actions only, Buy and Sell. They calculated their reward as percent change between current value and previous value. They tried to implement their DQN agent on the per minute data. I have implemented the same reward policy but I used daily data instead. I wanted to predict the data for longer investment strategy.

TODO:
1. Implement different state implementation than using raw data as states
2. Add total balance with the calculation of current portfolio value 

Long-term TODO:
Add sentiment analysis 
