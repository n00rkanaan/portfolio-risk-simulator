import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

class Portfolio:
    """
    The Portfolio class simulates a risk-managed equity portfolio over a historical price series.

    ticker_holdings is a dictionary mapping each ticker to a DataFrame tracking daily price,
    volume, total market value, and rolling 30-day risk metrics (Volatility, VaR, Max Drawdown,
    Sharpe, Sortino, and one custom metric).

    portfolio_history stores portfolio-level daily value, returns, and the same rolling metrics.

    pitch_map stores each pitch date and its proposed ticker, while pitch_log records every pitch decision made
    including the ticker, outcome, rationale, and recommended weight.

    Cash is tracked separately and must always satisfy a minimum buffer constraint.

    The simulation begins on day 30 to ensure a full 30-day window for all rolling metrics, and portfolio composition
    only changes on pitch dates via handle_pitch().
    """
    def __init__(self, universe_tickers, initial_tickers, initial_balance=100000):
        # Set of all tickers (including initial positions and potential pitches)
        self.tickers = universe_tickers
        # Tracks which tickers currently have a position
        self.active_tickers = set() #gets updated when buy or sell occurs but starts as empty
        # Tickers with initial positions
        self.initial_tickers = initial_tickers
        # Starting cash to invest with
        self.initial_balance = initial_balance

        # Dictionary containing dates of pitch dates
        self.pitch_map = {}

        # Data frame for recording pitch decisions
        self.pitch_log = pd.DataFrame(columns=[
            "Pitched Ticker",
            "Decision",
            "Reason",
            "Recommended Weight"
        ])
        self.pitch_log.index.name = "Date"

        # Data frame for recording Portfolio-level Metrics
        self.portfolio_history = pd.DataFrame(
            np.nan,
            index=pd.DatetimeIndex([], name="Date"),
            columns=["Portfolio Value",
                     "Portfolio Return",
                     "Max DD",
                     "Volatility",
                     "VaR",
                     "Sharpe",
                     "New Metric"],
            dtype=float
        )
        self.portfolio_history.index.name = "Date"

        # Data frame for recording Asset-level Metrics
        self.ticker_holdings = {}
        for ticker in self.tickers:
            df = pd.DataFrame(
                columns=["Price", "Unit Cost", "Volume", "Total Value",
                         "Max DD", "Volatility", "VaR", "Beta", "Sharpe", "Sortino", "New Metric"],
                index=pd.Index([], name="Date"))
            df["Price"] = 0.0
            df["Unit Cost"] = 0.0
            df["Volume"] = 0.0
            df["Total Value"] = 0.0
            df["Max DD"] = 0.0
            df["Volatility"] = 0.0
            df["VaR"] = 0.0
            df["Beta"] = 0.0
            df["Sharpe"] = 0.0
            df["Sortino"] = 0.0
            df["New Metric"] = 0.0
            self.ticker_holdings[ticker] = df

        # Cash Tracking
        self.cash = initial_balance
        self.cash_history = pd.DataFrame(columns=["Cash"])
        self.cash_history.index.name = "Date"

        # Benchmark Data
        self.market_prices = None

    # TO-DO
    def get_ticker_and_market_data(self, ticker_csv, pitch_csv, benchmark_csv):
        """
        Load all external data needed for the simulation.

        Requirements:
        1. Read ticker_data.csv and populate the "Price" column of each DataFrame in
           self.ticker_holdings with that ticker's adjusted close price series.
        2. Read benchmark_data.csv and store the SPY price series in self.market_prices.
        3. Read pitch_schedule.csv and populate self.pitch_map as {date: ticker},
           where each date is a Timestamp matching the shared date index.
        4. Set the index of self.pitch_log to the sorted dates in self.pitch_map.
        5. Re-initialize self.portfolio_history with the full date index now that
            it is known, so that prev_date lookups in the simulation never KeyError.
        """
        # Fill Steps 1-4 here:

        #1.
        ticker_prices_df = pd.read_csv(ticker_csv, index_col="Date", parse_dates=True)
        for ticker in self.tickers:
            df = self.ticker_holdings[ticker]
            #mk sure stock's df has same dates as price data - w/o this simulation would break
            df = df.reindex(ticker_prices_df.index)
            #add stock prices 
            df["Price"] = ticker_prices_df[ticker]
            #replace missing values w 0
            df["Volume"] = df["Volume"].fillna(0)
            df["Unit Cost"] = df["Unit Cost"].fillna(0)
            df["Total Value"] = df["Total Value"].fillna(0)
            self.ticker_holdings[ticker] = df #save updated dataframe
        
        #2. 
        market_prices_df = pd.read_csv(benchmark_csv, index_col="Date", parse_dates=True) 
        self.market_prices= market_prices_df["SPY"]
        
        #3. 
        pitch_df = pd.read_csv(pitch_csv)
        #convert dates to correct format yr/mm/dd
        pitch_df["Date"] = pd.to_datetime(pitch_df["Date"])
        for _, row in pitch_df.iterrows():
            #create dictionary so {date : ticker}
            self.pitch_map[row["Date"]] = row["Ticker"]

        #4. Set the index of self.pitch_log to the sorted dates in self.pitch_map.
        self.pitch_log= self.pitch_log.reindex(sorted(self.pitch_map.keys()))

       


        # Step 5: reinitialize portfolio_history with the full date index so that it is known when pulled on first day
        self.portfolio_history = pd.DataFrame(
            np.nan,
            index=ticker_prices_df.index,
            columns=["Portfolio Value",
                     "Portfolio Return",
                     "Max DD",
                     "Volatility",
                     "VaR",
                     "Beta",
                     "Sharpe",
                     "Sortino",
                     "New Metric"],
            dtype=float
        )
    #calculates total portfolio value today n daily return for a given day - how much is my portfolio worth today and how did it change compared to yesterday? 
    def update_holdings_and_portfolio_value(self, day):
        # Pull cash holdings
        portfolio_value = self.cash
        # Sum the Volume * Price of each ticker to calculate total value
        for ticker in self.tickers:
            volume = self.ticker_holdings[ticker].loc[day, "Volume"] #how many shares do i have
            price = self.ticker_holdings[ticker].loc[day, "Price"] #price of each share
            total_value = volume * price
            self.ticker_holdings[ticker].loc[day, "Total Value"] = total_value
            portfolio_value += total_value

        #Update Portfolio Value in Data Frame
        self.portfolio_history.loc[day, "Portfolio Value"] = portfolio_value

        # Gather Dates from Ticker Data Frame
        date_index = self.ticker_holdings[self.tickers[0]].index

        # Check this is not the first date and find the previous day and value
        if day != date_index[0]:
            prev_day = date_index[date_index.get_loc(day) - 1]
            prev_value = self.portfolio_history.loc[prev_day, "Portfolio Value"]

            #Calculate and Store Returns
            if pd.notna(prev_value) and prev_value != 0:
                self.portfolio_history.loc[day, "Portfolio Return"] = portfolio_value / prev_value - 1
            else:
                self.portfolio_history.loc[day, "Portfolio Return"] = 0.0
        else:
            #If this is the first day, Return = 0
            self.portfolio_history.loc[day, "Portfolio Return"] = 0.0

    def buy(self, ticker, count, date):
        """
        This function accepts the ticker, number of shares, and the date the transaction will be made.
        The holdings are all updated automatically.
        """
        # Gets the price of the ticker on the date
        price = self.ticker_holdings[ticker]["Price"].loc[date]

        # Finds the previous date
        prev_date = date
        row = 0
        if date != self.ticker_holdings[ticker]["Price"].index[0]:
            row = self.ticker_holdings[self.tickers[0]].index.get_loc(date)
            prev_date = self.ticker_holdings[self.tickers[0]].index[row-1]

        # Updates the portfolio holdings
        self.ticker_holdings[ticker].loc[date, "Volume"] = self.ticker_holdings[ticker].loc[prev_date, "Volume"] + count
        self.ticker_holdings[ticker].loc[date, "Unit Cost"] = ((self.ticker_holdings[ticker].loc[prev_date, "Unit Cost"] * self.ticker_holdings[ticker].loc[prev_date, "Volume"]) + (price * count)) / self.ticker_holdings[ticker].loc[date, "Volume"] 
        self.ticker_holdings[ticker].loc[date, "Total Value"] = self.ticker_holdings[ticker].loc[date, "Volume"] * self.ticker_holdings[ticker].loc[date, "Unit Cost"]
        self.cash -= (price * count)

        # Mark ticker as active
        self.active_tickers.add(ticker)

    def sell(self, ticker, count, date):
        """
        Similar function to buy(), however, processes a sell order
        """
        # Gets the price of the ticker on the date
        price = self.ticker_holdings[ticker]["Price"].loc[date]

        # Find the previous date
        prev_date = date
        if date != self.ticker_holdings[ticker]["Price"].index[0]:
            row = self.ticker_holdings[self.tickers[0]].index.get_loc(date)
            prev_date = self.ticker_holdings[self.tickers[0]].index[row - 1]

        # Updates portfolio holdings
        self.ticker_holdings[ticker].loc[date, "Unit Cost"] = self.ticker_holdings[ticker].loc[prev_date, "Unit Cost"]
        self.ticker_holdings[ticker].loc[date, "Volume"] = self.ticker_holdings[ticker].loc[prev_date, "Volume"] - count
        self.ticker_holdings[ticker].loc[date, "Total Value"] = self.ticker_holdings[ticker].loc[date, "Volume"] * self.ticker_holdings[ticker].loc[prev_date, "Unit Cost"]
        self.cash += (price * count)

        # Remove from active set if fully sold
        if self.ticker_holdings[ticker].loc[date, "Volume"] <= 0:
            self.active_tickers.discard(ticker)

    def simulate(self):
        """
        This function processes the time series data of adjusted closed prices via a for loop
        to simulate trading days taking place
        Data for every stock is downloaded from the csvs using the get_ticker_and_market_data method,
        and cleaned into a data frames for easier processing
        The simulation begins on day 30 to ensure sufficient data for all 30-day rolling risk
        metrics from the start
        """

        # Download data for all tickers and market data
        self.get_ticker_and_market_data("ticker_data.csv", "pitch_schedule.csv", "benchmark_data.csv")

        # Buy initial holdings on Day 30 (equal value for each stock and cash)
        start_day_index = 30
        first_day = self.ticker_holdings[self.tickers[0]].index[start_day_index]
        for stock in self.tickers:
            self.ticker_holdings[stock].loc[first_day, "Volume"] = 0.0

        for stock in self.initial_tickers:
            price_share = self.ticker_holdings[stock]["Price"].iloc[start_day_index]
            allocation = self.initial_balance / len(self.initial_tickers)
            shares = int(allocation / price_share)
            self.buy(stock, shares, first_day)

        # Compute value from current volumes and log
        self.update_holdings_and_portfolio_value(first_day)
        self.portfolio_history.loc[first_day, "Portfolio Return"] = 0.0
        self.calculate_metrics(first_day)
        self.cash_history.loc[first_day] = self.cash

        # Daily simulation from day 31
        prev_day = first_day
        for day in self.ticker_holdings[self.tickers[0]].index[start_day_index+1:]:
            # Carry forward volumes from previous day
            for ticker in self.tickers:
                self.ticker_holdings[ticker].loc[day, "Volume"] = self.ticker_holdings[ticker].loc[prev_day, "Volume"]
                self.ticker_holdings[ticker].loc[day, "Unit Cost"] = self.ticker_holdings[ticker].loc[prev_day, "Unit Cost"]

            # Compute value and metrics before any pitch
            self.update_holdings_and_portfolio_value(day)
            self.calculate_metrics(day)

            # Handle pitch (may execute trades via buy/sell)
            self.handle_pitch(day)

            # If a pitch occurred, recompute metrics and portfolio data from updated volumes
            if day in self.pitch_map:
                self.update_holdings_and_portfolio_value(day)
                self.calculate_metrics(day)
            self.cash_history.loc[day] = self.cash # Update Cash
            prev_day = day #Move to the next day

        #Display data at end of simulation
        self.display_data()

    # TO-DO
    #every day look back at prev 30 days n calc risk metrics for each stock and the whole portfolio
    def calculate_metrics(self,date):
        """
        Calculate previous 30-day risk metrics.

        Requirements:
        1. Asset-level metrics:
           For each ticker in self.ticker_holdings, compute previous 30-day metrics
           (VaR, Volatility, Max Drawdown, Beta, Sharpe, Sortino and one custom metric) using that
           ticker's return series. Store results in the ticker's DataFrame.
        2. Portfolio-level metrics:
           Using self.portfolio_history["Portfolio Return"], compute the same rolling
           30-day metrics at the portfolio level and store them in self.portfolio_history.

        Notes:
        - Metrics should be defined "as of" each day using the prior 30 trading days.
        - VaR method: Historical simulation at 95% confidence — the 5th percentile of
          the empirical 30-day return distribution. No parametric distribution is assumed.
        - Use self.market_prices for any benchmark needs
        """

        """
        New Metric Explanation:
        Downside vol measures negative fluctuations of returns and focuses specifically on loss risk. I chose it since it differs from standard vol where it doesnt focus on the positive price movements. This custom metric complements sortino and sharpe ratio by showing the extent of negative price movements.
        """

        #1. 
      

        #go thru each ticker
        for ticker in self.tickers:
            prices = self.ticker_holdings[ticker]["Price"].loc[:date]
            #convert prices to daily returns
            returns = prices.pct_change().dropna()
            prior_thirty_days = returns.iloc[-30:]
            #not enough data, skip
            if len(prior_thirty_days) < 30:
                continue
            #mk sure stock returns n market returns line up on same dates for beta calculation
            market_returns = self.market_prices.pct_change().loc[prior_thirty_days.index].dropna()
            #mk stock returns match exactly cleaned market dates
            prior_thirty_days_aligned = prior_thirty_days.loc[market_returns.index]

            #VaR = 95% Confidence Var, so the 5th percentile will be used to show the losses - worst 5% outcome
            var = np.percentile(prior_thirty_days, 5)
           
            #Vol
            vol = prior_thirty_days.std()
           
            #Max Drawdown = (peak val - trough val) / peak val - worst loss from a high point
            peak_val = prices.iloc[-30:].cummax()
            drawdown = (prices.iloc[-30:] - peak_val) / peak_val
            max_drawdown = drawdown.min()

            #Beta = Cov(stock, market) / var(market)
            beta = (np.cov(prior_thirty_days_aligned, market_returns)[0][1] / np.var(market_returns)
                    if len(market_returns) > 1 else np.nan)

            #sharpe = mean / sd(total) - higher sharpe is a better risk adj return
            sharpe = prior_thirty_days.mean() / vol if vol != 0 else 0
           
            #sortino = Return of port - risk free rate(0) / sd(downside) - higher sortino is better bc bettr risk-adjusted performance
            downside_vol = prior_thirty_days[prior_thirty_days < 0].std()
            sortino = prior_thirty_days.mean() / downside_vol if downside_vol != 0 else 0
            
            #custom metric - Downside volatility (def above)
            custom  = downside_vol #basically written and used above

            #Store results in the ticker's DataFrame for that day. VaR, Volatility, Max Drawdown, Beta, Sharpe, Sortino 
            self.ticker_holdings[ticker].loc[date, "VaR"] = var
            self.ticker_holdings[ticker].loc[date, "Volatility"] = vol
            self.ticker_holdings[ticker].loc[date, "Max DD"] = max_drawdown
            self.ticker_holdings[ticker].loc[date, "Beta"] = beta
            self.ticker_holdings[ticker].loc[date, "Sharpe"] = sharpe
            self.ticker_holdings[ticker].loc[date, "Sortino"] = sortino
            self.ticker_holdings[ticker].loc[date, "New Metric"] = custom

        #2. Portfolio-level metrics using portfolio return series
        port_returns = self.portfolio_history["Portfolio Return"].loc[:date].dropna()
        prior_thirty_days_port = port_returns.iloc[-30:]
        #variance n covariance need at least 2 data points
        if len(prior_thirty_days_port) < 2:
            return

        market_returns_port = self.market_prices.pct_change().loc[prior_thirty_days_port.index].dropna()
        prior_thirty_days_port_aligned = prior_thirty_days_port.loc[market_returns_port.index]

        #VaR = 95% Confidence Var, so the 5th percentile will b used to show the losses
        port_var = np.percentile(prior_thirty_days_port, 5)

        #Vol
        port_vol = prior_thirty_days_port.std()

        #Max Drawdown = (peak val - trough val) / peak val
        port_values = self.portfolio_history["Portfolio Value"].loc[:date].dropna().iloc[-30:]
        port_peak_val = port_values.cummax()
        port_drawdown = (port_values - port_peak_val) / port_peak_val
        port_max_drawdown = port_drawdown.min()

        #Beta = Cov(portfolio, market) / var(market)
        port_beta = (np.cov(prior_thirty_days_port_aligned, market_returns_port)[0][1] / np.var(market_returns_port)
                     if len(market_returns_port) > 1 else np.nan)

        #sharpe = mean / sd(total) - higher sharpe is a better risk adj return
        port_sharpe = prior_thirty_days_port.mean() / port_vol if port_vol != 0 else 0

        #sortino = Return of port - risk free rate / sd(downside)
        port_downside_vol = prior_thirty_days_port[prior_thirty_days_port < 0].std()
        port_sortino = prior_thirty_days_port.mean() / port_downside_vol if port_downside_vol != 0 else 0

        #custom metric - Downside volatility
        port_custom = port_downside_vol

        #Store results in portfolio_history
        self.portfolio_history.loc[date, "VaR"] = port_var
        self.portfolio_history.loc[date, "Volatility"] = port_vol
        self.portfolio_history.loc[date, "Max DD"] = port_max_drawdown
        self.portfolio_history.loc[date, "Beta"] = port_beta
        self.portfolio_history.loc[date, "Sharpe"] = port_sharpe
        self.portfolio_history.loc[date, "Sortino"] = port_sortino
        self.portfolio_history.loc[date, "New Metric"] = port_custom


    # TO-DO
    def handle_pitch(self, date, cash_buffer=0.01):
        #only on pitch dates
        if date not in self.pitch_map:
            return 
        #pull ticker being pitched n its data (prices n metrics)
        ticker = self.pitch_map[date]
        df = self.ticker_holdings[ticker]

        vol = df.loc[date, "Volatility"] #total risk
        var = df.loc[date, "VaR"] #worst case loss
        sharpe = df.loc[date, "Sharpe"] #risk-adjusted return
        downside_vol = df.loc[date, "New Metric"] #negative risk 

        #variable to keep track of how many risk metric conditions are met
        conditions_met = 0 

        #thresholds for risk metrics
        if pd.notna(sharpe) and sharpe > 0: #positive return per unit risk
            conditions_met+=1
        if pd.notna(vol) and vol < 0.02: #low vol needed
            conditions_met+=1
        if pd.notna(downside_vol) and downside_vol < 0.015: #low downside risk
            conditions_met+=1
        if pd.notna(var) and var > -0.03: #limited losses
            conditions_met+=1
        
        #how many conditions met
        if conditions_met >= 3:
            decision = "Stock Accepted"
            reason = (f"Passed {conditions_met}/4 risk checks: "
                      f"Sharpe={sharpe:.3f}, Vol={vol:.4f}, VaR={var:.4f}, DnVol={downside_vol:.4f}")
            
            self.active_tickers.add(ticker)
            portfolio_value = self.portfolio_history.loc[date, "Portfolio Value"]
        
            #make all stocks be equally weighted
            target_weight = 1 / len(self.active_tickers)
            #see how much i can invest via cash
            investable = portfolio_value * (1 - cash_buffer)
            #rebalance all stocks
            for stock in list(self.active_tickers):
                price = self.ticker_holdings[stock].loc[date, "Price"]
                shares = self.ticker_holdings[stock].loc[date, "Volume"]
                #if diff is pos we buy; else we sell
                current_value = shares * price
                target_value = investable * target_weight
                difference = target_value - current_value
                #turn dollar diff to number of shares
                shares_to_trade = int(abs(difference) / price)

                if shares_to_trade > 0: 
                    if difference > 0:
                        # Enforce cash buffer before buying
                        cost = shares_to_trade * price
                        #dont break minimum cash rule
                        if self.cash - cost < portfolio_value * cash_buffer:
                            #then we adjust so we only buy what we can afford
                            shares_to_trade = int((self.cash - portfolio_value * cash_buffer) / price)
                        if shares_to_trade > 0:
                            self.buy(stock, shares_to_trade, date)
                    else:
                        # Enforce long-only: we never sell more than current shares (what we own)
                        shares_to_trade = min(shares_to_trade, int(shares))
                        if shares_to_trade > 0:
                            self.sell(stock, shares_to_trade, date)
        
        else:
            decision = "Stock Rejected"
            reason = (f"Only {conditions_met}/4 risk checks passed: "
                      f"Sharpe={sharpe:.3f}, Vol={vol:.4f}, VaR={var:.4f}, DnVol={downside_vol:.4f}")
            target_weight = 0

        self.pitch_log.loc[date, "Pitched Ticker"] = ticker
        self.pitch_log.loc[date, "Decision"] = decision
        self.pitch_log.loc[date, "Reason"] = reason
        self.pitch_log.loc[date, "Recommended Weight"] = target_weight


    #     """
    #     Evaluate a pitched stock and, if approved, rebalance the portfolio to include it.
    #     Only execute trades on pitch days, return immediately on all other days.

    #     Requirements:
    #     1. Approve or deny:
    #         Decide whether to add the ticker using at least three risk metrics as
    #         criteria, including your custom metric. Define your own thresholds.
    #         If denied, log the decision with an explanation.
    #     2. Determine weights:
    #         Choose a primary portfolio metric and a specific target value for it.
    #         Select the weight for the new ticker and the weights for all existing
    #         holdings with the goal of reaching this target. You may also choose to
    #         rebalance the existing stocks on pitch dates even when the stock is denied.
    #         Justify your weighting approach in a comment.
    #         NOTE: You can only use historical data to make this decision, do not
    #         use future data to decide weights because this would be impossible in practice
    #     3. Execute trades:
    #         Use self.buy() and self.sell() to rebalance each holding to its target
    #         share count. self.cash must not fall below cash_buffer * portfolio_value.
    #     4. Log the pitch:
    #         Record Ticker, Decision, Rationale, and Recommended Weight in self.pitch_log.

    #     Additionally, after executing all trades on a pitch date, you are responsible for ensuring that:
    #     - All positions remain long-only (no negative share counts)
    #     - Cash remains non-negative and satisfies the required cash buffer
    #     - Total invested capital does not exceed available capital

    #     If your proposed rebalance violates any of these constraints, you must adjust or reject the pitch.
    #     """
    # """
    # - Approval/Denial Criteria Justification:
    # I decided to use Sharpe, Volatility, VaR, and Downside Volatility as my four criteria since
    # they each capture a different angle of risk. Sharpe tells you if the return is actually worth
    # the risk, Volatility shows how much the price is moving around in general, VaR gives a sense
    # of how bad a single bad day could get, and Downside Vol (my custom metric) focuses only on
    # the negative days specifically. I set the thresholds at Sharpe > 0, Vol < 0.02, VaR > -0.03,
    # and Downside Vol < 0.015 since those felt like reasonable cutoffs for a diversified portfolio.
    # stocks that fail these are just too volatile or risky to add.
    
    # - Weight Distribution Justification: 
    # When a stock gets accepted I rebalance everything to equal weights across all active positions,
    # keeping 1% of the portfolio as a cash buffer. I went with equal weighting because it's simple,
    # avoids putting too much into any one stock, and doesn't require any assumptions about future
    # returns. I don't rebalance when a stock gets rejected if it didn't pass the risk checks as
    # there's no reason to shuffle the existing holdings around and rack up unnecessary trades.
    
    
    # """

    # TO-DO
    def display_data(self):

        """
        Produce a dashboard showing portfolio performance and risk over the simulation.

        Requirements:
        1. Plot portfolio value over time vs. the SPY benchmark (self.market_prices),
           normalized to the same starting value.
        2. Plot daily portfolio returns over time.
        3. Plot at least two rolling risk metrics over time (e.g. Volatility and VaR,
           or Sharpe and your custom metric).
           Also mark the target value of your selected risk metrics used for weighting
        4. On all plots, mark each pitch date with a vertical dashed line:
             - Green = pitch was APPROVED
             - Red = pitch was DENIED
        5. Print or display a table of the current holdings as of the last simulation
           day, showing at minimum: Ticker, Shares, Price, Total Value, Weight (%).
        6. Print the pitch_log DataFrame showing all pitch decisions.
        Use self.market_prices for the benchmark series.
        """

        """
        Comment on the performance of your decisions (Did you achieve your risk targets?, How did your risk compare to
        the benchmark?, etc.)
        Overall I think the risk framework worked pretty well. The main goal was to keep volatility
        and downside risk controlled, and looking at the rolling metrics the portfolio generally stayed
        below the thresholds I set. Sharpe stayed positive for most of the period and Downside Vol
        rarely spiked too high. The rejections made sense too, TSLA and NFLX were both going through
        really choppy periods when they were pitched so keeping them out was the right call.

        Compared to SPY the portfolio ended up with good returns overall which isn't surprising since
        we were being pretty selective and holding cash. The drawdowns were also smaller during
        the rougher stretches in 2022 which is kind of the whole point of managing risk this way.
        If I were to change anything I'd maybe loosen the Vol threshold slightly since it ended up
        rejecting a few stocks that probably would have been fine long term, but the conservative
        approach did what it was supposed to do. 

        """

        # Trim to simulation window (day 30 onward)
        ph = self.portfolio_history.dropna(subset=["Portfolio Value"])
        sim_start = ph.index[0]

        # SPY normalised to portfolio starting value
        spy = self.market_prices.loc[sim_start:]
        spy_normalized_to_same_starting_val = spy / spy.iloc[0] * ph["Portfolio Value"].iloc[0]

        # Helper to add pitch-date vertical lines (green=approved, red=denied)
        def add_pitch_lines(ax):
            for date in self.pitch_map:
                if date >= sim_start:
                    decision = self.pitch_log.loc[date, "Decision"] if date in self.pitch_log.index else ""
                    color = "green" if decision == "Stock Accepted" else "red"
                    ax.axvline(x=date, linestyle="--", linewidth=0.8, color=color, alpha=0.7)

        fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
        fig.suptitle("Portfolio Risk Dashboard", fontsize=15, fontweight="bold")

        # Plot 1: Portfolio Value vs SPY
        ax = axes[0]
        ax.plot(ph.index, ph["Portfolio Value"], label="Portfolio", linewidth=1.5, color="steelblue")
        ax.plot(spy_normalized_to_same_starting_val.index, spy_normalized_to_same_starting_val,
                label="SPY Benchmark", linewidth=1.5, color="darkorange", linestyle="--")
        add_pitch_lines(ax)
        ax.set_ylabel("Value ($)")
        ax.set_title("Portfolio Value vs SPY Benchmark")
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # Plot 2: Daily Portfolio Returns
        ax = axes[1]
        ax.plot(ph.index, ph["Portfolio Return"] * 100, linewidth=0.8, color="steelblue", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        add_pitch_lines(ax)
        ax.set_ylabel("Daily Return (%)")
        ax.set_title("Daily Portfolio Returns")

        # Plot 3: Sharpe and Volatility with threshold lines
        ax = axes[2]
        ax.plot(ph.index, ph["Sharpe"],     label="Sharpe (30d)",     color="purple", linewidth=1.2)
        ax.plot(ph.index, ph["Volatility"], label="Volatility (30d)", color="teal",   linewidth=1.2)
        ax.axhline(0,    color="purple", linestyle=":", linewidth=0.8, label="Sharpe target = 0")
        ax.axhline(0.02, color="teal",   linestyle=":", linewidth=0.8, label="Vol threshold = 0.02")
        add_pitch_lines(ax)
        ax.set_ylabel("Metric value")
        ax.set_title("Rolling 30-day Sharpe Ratio and Volatility")
        ax.legend(loc="upper left", fontsize=8)

        # Plot 4: VaR and Custom Metric (Downside Volatility)
        ax = axes[3]
        ax.plot(ph.index, ph["VaR"],        label="VaR 95% (30d)",      color="crimson",    linewidth=1.2)
        ax.plot(ph.index, ph["New Metric"], label="Downside Vol (30d)", color="darkorange", linewidth=1.2)
        ax.axhline(-0.03, color="crimson",    linestyle=":", linewidth=0.8, label="VaR threshold = -0.03")
        ax.axhline(0.015, color="darkorange", linestyle=":", linewidth=0.8, label="DnVol threshold = 0.015")
        add_pitch_lines(ax)
        ax.set_ylabel("Metric value")
        ax.set_title("Rolling 30-day VaR (95%) and Downside Volatility (Custom Metric)")
        ax.legend(loc="upper left", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # Shared legend for pitch lines
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="green", linestyle="--", linewidth=1, label="Pitch Approved"),
            Line2D([0], [0], color="red",   linestyle="--", linewidth=1, label="Pitch Denied"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, 0.01))

        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig("portfolio_dashboard.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Dashboard saved to portfolio_dashboard.png")

        # Holdings table as of last simulation day
        last_day = ph.index[-1]
        total_val = ph.loc[last_day, "Portfolio Value"]
        print(f"\n{'='*65}")
        print(f"HOLDINGS as of {last_day.date()}   (Total Portfolio Value: ${total_val:,.2f})")
        print(f"{'='*65}")
        print(f"{'Ticker':<8} {'Shares':>8} {'Price':>10} {'Total Value':>14} {'Weight %':>10}")
        print(f"{'-'*65}")
        for ticker in sorted(self.active_tickers):
            shares = self.ticker_holdings[ticker].loc[last_day, "Volume"]
            price  = self.ticker_holdings[ticker].loc[last_day, "Price"]
            val    = shares * price
            wgt    = val / total_val * 100 if total_val else 0
            print(f"{ticker:<8} {shares:>8.0f} {price:>10.2f} {val:>14.2f} {wgt:>9.1f}%")
        print(f"{'Cash':<8} {'':>8} {'':>10} {self.cash:>14.2f} {self.cash/total_val*100:>9.1f}%")
        print(f"{'-'*65}")
        print(f"{'TOTAL':<8} {'':>8} {'':>10} {total_val:>14.2f} {'100.0':>9}%")

        # Pitch log
        print(f"\n{'='*65}")
        print("PITCH LOG")
        print(f"{'='*65}")
        pd.set_option("display.max_colwidth", 80)
        pd.set_option("display.width", 120)
        print(self.pitch_log.to_string())

        # Final portfolio risk snapshot
        print(f"\n{'='*65}")
        print("PORTFOLIO RISK METRICS — last 5 days")
        print(f"{'='*65}")
        cols = ["Portfolio Value", "Portfolio Return", "Sharpe", "Volatility",
                "VaR", "Sortino", "New Metric", "Max DD", "Beta"]
        print(ph[cols].tail(5).to_string())


port = Portfolio(
    universe_tickers=[
    "AAPL", "MSFT", "JPM", "JNJ", "XOM", "UPS", "HD", "PG", "BAC", "NVDA",
    "AVGO", "CAKE", "WMT", "DIS", "NFLX", "AMZN", "GS", "PFE", "CVX", "TSLA"
    ],
    initial_tickers=[
         "AAPL", "MSFT", "JPM",  "JNJ",  "XOM"
    ],
    initial_balance=100000
)
port.simulate()

print("\n")