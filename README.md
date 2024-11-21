# ✨ Velora Pool Data Fetcher ✨  

## Overview  

The **Velora Pool Data Fetcher** is a Python-based application designed to retrieve and process data from Uniswap V3 pools. It stores the fetched data in a PostgreSQL database with the TimescaleDB extension for efficient time-series handling. The application is containerized using Docker and includes a CI/CD pipeline for automated testing and deployment.  

---

## 🌟 Features  

- **Efficient Batch Processing**: Processes data in batches of 80 tokens to ensure optimal performance.  
- **Comprehensive Uniswap Data Retrieval**: Supports fetching data for events like swaps, mints, burns, and collects.  
- **Database Integration**: Leverages PostgreSQL with TimescaleDB for scalable and efficient storage.  
- **Signal Generation**: Automatically generates and stores signals based on the fetched data.  
- **CI/CD Pipeline**: Includes a GitHub Actions workflow for automated testing and deployment.  

---

## 🛠️ Prerequisites  

Before you begin, ensure you have the following installed:  

- **Python 3.10 or higher**  
- **PostgreSQL with TimescaleDB extension**  

---

## 🚀 Setup  

### Running Ethereum node [Optional]

You can run your own Ethereum node locally using the following command:

```bash
docker compose up -d geth prysm
```

Please note that this process may take a significant amount of time to fetch all the necessary data.

If you have an alternative Ethereum node that you’d like to use, you can specify it in your `.env` file.

---

### Running PostgreSQL

Run PostgreSQL timescale db with docker compose

```bash
docker compose up -d timescaledb
```

If you are already running postgres server, you can specify it in your `.env` file.

---

### Running Fetcher

1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/nestlest/velora-pool-data-fetcher.git  
   cd velora-pool-data-fetcher  
   ```  

2. **Create and configure the `.env` file**:  
   Copy the provided `.env.example` file and fill in the required environment variables.  
   ```bash  
   cp .env.example .env  
   ```  

3. **Install Virtual Environment**
   It is recommended to use virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python dependencies**:  
   ```bash  
   pip3 install -r requirements.txt  
   ```  

---

## 🌐 Usage  

1. **Run the application**:  
   ```bash  
   pm2 start --name velora-pool-data-fetcher python main.py  
   ```  

2. **Access the PostgreSQL database**:  
   The database is exposed on the port specified in your `.env` file. Use any PostgreSQL client to connect and query the data.  

---

## 🔧 Configuration  

The application uses environment variables for configuration, loaded from the `.env` file. Key variables include:  

- `POSTGRES_USER`  
- `POSTGRES_PASSWORD`  
- `POSTGRES_DB`  
- `POSTGRES_HOST`  
- `POSTGRES_PORT`  
- `ETHEREUM_RPC_NODE_URL`  

---

## 🤝 Contributing  

We welcome contributions to improve **Velora Pool Data Fetcher**! Here's how you can contribute:  

1. 🍴 **Fork the repository**: Create a copy on your GitHub account.  
2. 🛠️ **Make updates or add features**: Work on your changes locally.  
3. 📤 **Submit a pull request (PR)**: Propose your updates to the main repository.  

### Notes:  
- 🔍 All PRs are reviewed before merging.  
- 💡 Suggestions for enhancements are always welcome!  

**Your contributions help us grow! 🌱**  

---

## 📜 License  

This project is licensed under the MIT License. See the `LICENSE` file for more details.  

---

## 📬 Contact  

For any questions or issues, please open an issue on the GitHub repository.  
Join our community if you have any questions: [Join Our Velora Channel](https://discord.com/channels/941362322000203776/1301167504504127609)
