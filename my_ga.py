import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint, choice

# Διαβάζουμε το CSV αρχείο
data = pd.read_csv("iphi2802.csv", encoding="utf-8")

# Φιλτράρουμε τις επιγραφές που έχουν το ίδιο region_main_id ή region_main == 'Greater Syria and the East'
region_id = 1693
filtered_data = data[(data['region_main_id'] == region_id) | (data['region_main'] == 'Greater Syria and the East')]

# Ανάκτηση του πεδίου text για τις επιγραφές
X_text_tfidf = filtered_data['text'].tolist()

# Αρχικοποίηση του Vectorizer με tf-idf
vectorizer = TfidfVectorizer(max_features=1678)

# Μετατροπή του κειμένου σε διανύσματα tf-idf
X = vectorizer.fit_transform(X_text_tfidf)

# Υπολογισμός του μεγέθους του λεξικού
vocabulary_size = X.shape[1]

# Ανάκτηση λεξικού token σε μια μεταβλητη
vocabulary = vectorizer.get_feature_names_out()

# Ενημέρωση του λεξικού ώστε να περιέχει μόνο τα πρώτα 1678 tokens
if len(vocabulary) > 1678:
    vocabulary = vocabulary[:1678]

# Δημιουργία αντιστοίχησης λέξεων σε ακέραιους και αντίστροφα
word_to_int = {word: i for i, word in enumerate(vocabulary)}
int_to_word = {i: word for i, word in enumerate(vocabulary)}

# υπολογισμος αποστασης manhattan μεταξυ δυο κειμενων
def manhattan_distance(text1, text2):
    distance = sum(abs(ord(c1) - ord(c2)) for c1, c2 in zip(text1, text2))
    return distance
# τα k πιο κοντινα κειμενα
def find_top_k_closest(target_text, k=5):
    distances = [(index, manhattan_distance(target_text, text)) for index, text in enumerate(X_text_tfidf)]
    sorted_distances = sorted(distances, key=lambda x: x[1])
    top_k = sorted_distances[:k]
    return top_k

# βασικη συναρτηση ανακτησης χαμενων επιγραφων
def restore_missing_words(target_text, max_generations, population_size, crossover_rate, mutation_rate):
    target_vector = vectorizer.transform([target_text])
    # υπολογισμος αποδοσης ατομου με χρηση cosine similarity
    def fitness(individual, target_vector):
        total_similarity = 0
        for _ in range(len(individual)):
            individual_text = ' '.join(int_to_word[i] for i in individual)
            individual_vector = vectorizer.transform([individual_text])
            similarity = cosine_similarity(target_vector, individual_vector)
            total_similarity += similarity[0][0]
        return total_similarity

    def random_individual():
        return [np.random.choice(len(vocabulary)) for _ in range(2)]
     #διασταυρωση με χρηση μονου σημειου
    def crossover(parent1, parent2):
        if randint(0,100) < crossover_rate * 100:
            crossover_point = np.random.randint(1, len(parent1))
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2
     #μεταλλαξη με χρηση ελιτισμου
    def mutate(individual, mutation_rate):
        for i in range(len(individual)):
          if individual != best_individual: 
            if randint(0, 100) < mutation_rate * 100:
                individual[i] = np.random.choice(len(vocabulary))
        return individual
     #επιλογη με χρηση τουρνουα
    def tournament_selection(population, fitness_scores, k=3):
        selection_ix = np.random.choice(len(population))
        for _ in range(k - 1):
            ix = np.random.choice(len(population))
            if fitness_scores[ix] > fitness_scores[selection_ix]:
                selection_ix = ix
        return population[selection_ix]
  
    population = [random_individual() for _ in range(population_size)]
    best_fitness_history = []

    for generation in range(max_generations):
        fitness_scores = [fitness(individual, target_vector) for individual in population]
        # ευρεση καλυτερου ατομου
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        best_fitness = fitness_scores[best_index]
        best_fitness_history.append(best_fitness)

        if best_fitness == 1.0:
            return best_individual, best_fitness, generation, best_fitness_history

        selected_parents = [tournament_selection(population, fitness_scores) for _ in range(population_size)]

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(len(selected_parents), size=2, replace=False)
            child1, child2 = crossover(selected_parents[parent1], selected_parents[parent2])
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population[:population_size]

    return best_individual, best_fitness, max_generations, best_fitness_history

target_text = "ουαλεριον ιουλιανον τον κρατιστον επιτροπον των σεβαστων η μητροπολις και μητροκολωνια τον εαυτης ευεργετην δια αυρηλιων θεοδορου και αριστειδου στρατηγων."

num_experiments = 12
max_generations = 49
population_size = 200
crossover_rate = 0.1
mutation_rate = 0.01

all_fitness_histories = []

for _ in range(num_experiments):
    restored_text_indices, best_fitness, generation_count, best_fitness_history = restore_missing_words(target_text, max_generations=max_generations, population_size=population_size, crossover_rate=crossover_rate, mutation_rate=mutation_rate)
    restored_text = ' '.join(int_to_word[i] for i in restored_text_indices)
    all_fitness_histories.append(best_fitness_history)
    print("Restored Text:", restored_text)
    print("Best Fitness:", best_fitness)
    print("Generations:", generation_count)

# Υπολογισμός της μέσης τιμής της fitness σε κάθε γενιά
average_fitness_history = np.mean(all_fitness_histories, axis=0)

# Σχεδίαση της καμπύλης εξέλιξης
plt.plot(range(max_generations), average_fitness_history)
plt.xlabel('Generations')
plt.ylabel('Average Best Fitness')
plt.title('Evolution of Best Fitness Over Generations')
plt.show()

# Υπολογισμός του μέσου αριθμού γενεών και του μέσου καλύτερου fitness
average_generations = np.mean([len(i) for i in all_fitness_histories])
average_best_fitness = np.mean([np.max(i) for i in all_fitness_histories])

print("Average Number of Generations:", average_generations)
print("Average Best Fitness:", average_best_fitness)
print("Number of Tokens:", len(vocabulary)) 

# Υπολογισμός των top-5 κοντινότερων επιγραφών στο target_text
top_5_closest = find_top_k_closest(target_text, k=5)

# Εκτύπωση των top-5 κοντινότερων επιγραφών
print("Top 5 Closest Texts:")
for index, distance in top_5_closest:
    print(f"Manhattan Distance: {distance}, Text: {X_text_tfidf[index]}")

