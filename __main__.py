from CustomRandomForestClassifier import CustomRandomForestClassifier

custom_rf_classifier = CustomRandomForestClassifier()

# Load and preprocess data
custom_rf_classifier.load_and_preprocess_data('./DB/LyricsFirstPeriod.csv', './DB/LyricsSecondPeriod.csv')

# Find the best max_features value
best_accuracy, best_max_features = custom_rf_classifier.find_best_max_features()
print(f'Best max_features: {best_max_features} with accuracy: {best_accuracy}')

# Train the final model with the best max_features
accuracy, report = custom_rf_classifier.train_final_model(best_max_features)
print(f'Final Model Accuracy: {accuracy}')
print(report)

# Save the model and vectorizer
custom_rf_classifier.save_model('eminem_lyrics_model_rf.pkl', 'tfidf_vectorizer_rf.pkl')
