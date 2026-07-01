"""This file is meant to run any of my package code."""


# config = Config()
# rules = config.get_preprocessing_rules()

# data = Data()
# data.fetch_and_store_forecasts()
# data.read_raw()
# dataset = data.generate_dataset(location_name="aachen")

# model = LinearUncertaintyModel(config)
# feature_columns = [
#     "day_of_year",
#     "delta_days",
#     "temperature_2m_max",
#     "temperature_2m_min",
#     "precipitation_sum",
#     "sunshine_duration",
#     "wind_direction_10m_dominant",
#     "wind_speed_10m_max",
#     "wind_gusts_10m_max",
#     "wind_speed_10m_mean",
#     "precipitation_probability_mean"
# ]
# target_columns = [
#     "abs_diff__temperature_2m_max",
#     "abs_diff__temperature_2m_min",
#     "abs_diff__precipitation_sum",
#     "abs_diff__sunshine_duration",
#     "abs_diff__wind_speed_10m_mean",
#     "abs_diff__precipitation_probability_mean"
# ]
# model.fit(dataset, feature_columns, target_columns)

# summer_day = {
#     "day_of_year": 150,
#     "delta_days": 3.0,  # 3 days between forecasted_on and forecast_for
#     "temperature_2m_max": 25.0,
#     "temperature_2m_min": 15.0,
#     "precipitation_sum": 0.0,
#     "sunshine_duration": 8.0,
#     "wind_direction_10m_dominant": 180.0,
#     "wind_speed_10m_max": 20.0,
#     "wind_gusts_10m_max": 35.0,
#     "wind_speed_10m_mean": 10.0,
#     "precipitation_probability_mean": 0.1
# }

# winter_blizzard = {
#     "day_of_year": 15,          # Mid-January
#     "delta_days": 5.0,          # High look-ahead horizon (higher error expected)
#     "temperature_2m_max": -5.0,
#     "temperature_2m_min": -12.0,
#     "precipitation_sum": 25.0,  # Heavy snowfall/liquid equivalent
#     "sunshine_duration": 0.0,   # Overcast
#     "wind_direction_10m_dominant": 340.0, # North-Northwest arctic blast
#     "wind_speed_10m_max": 45.0,
#     "wind_gusts_10m_max": 75.0, # High gust volatility
#     "wind_speed_10m_mean": 30.0,
#     "precipitation_probability_mean": 0.95
# }

# stable_summer_high = {
#     "day_of_year": 200,         # Late July
#     "delta_days": 4.0,
#     "temperature_2m_max": 32.0,
#     "temperature_2m_min": 18.0,
#     "precipitation_sum": 0.0,   # No rain
#     "sunshine_duration": 14.5,  # Maximum clear sky duration
#     "wind_direction_10m_dominant": 90.0, # Stable easterly breeze
#     "wind_speed_10m_max": 8.0,
#     "wind_gusts_10m_max": 12.0,
#     "wind_speed_10m_mean": 4.0,
#     "precipitation_probability_mean": 0.05
# }

# nowcast_calm_spring = {
#     "day_of_year": 100,         # April
#     "delta_days": 0.0,          # Target: Near-zero uncertainty
#     "temperature_2m_max": 14.0,
#     "temperature_2m_min": 6.0,
#     "precipitation_sum": 0.0,
#     "sunshine_duration": 9.0,
#     "wind_direction_10m_dominant": 220.0,
#     "wind_speed_10m_max": 10.0,
#     "wind_gusts_10m_max": 15.0,
#     "wind_speed_10m_mean": 5.0,
#     "precipitation_probability_mean": 0.10
# }

# nowcast_autumn_storm = {
#     "day_of_year": 290,         # October
#     "delta_days": 0.0,          # Target: Near-zero uncertainty despite volatility
#     "temperature_2m_max": 12.0,
#     "temperature_2m_min": 4.0,
#     "precipitation_sum": 40.0,  # Heavy active rain
#     "sunshine_duration": 1.0,
#     "wind_direction_10m_dominant": 270.0, # Strong westerly gale
#     "wind_speed_10m_max": 55.0,
#     "wind_gusts_10m_max": 90.0,
#     "wind_speed_10m_mean": 35.0,
#     "precipitation_probability_mean": 0.99
# }

# test_cases = [summer_day, winter_blizzard, stable_summer_high, nowcast_calm_spring, nowcast_autumn_storm]
# df_test_cases = pd.DataFrame(test_cases)
# uncertainties = model.predict(df_test_cases)
# for col in uncertainties.iterrows():
#     print(col)

# for item in model.processed_feature_columns:
#     print(item)
# print("\n\n\n")
# print(model.X.head(20))
# print("\n\n\n")
# print(model.Y)
# print("\n\n\n")
# print(model.preprocessor.rules)

from fue import Forecast

F = Forecast()
F.fetch_forecast(location_name="berlin", forecast_days=14)
print(F.forecast)
