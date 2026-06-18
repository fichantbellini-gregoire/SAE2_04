import numpy as np
import numpy.linalg as lag
import pandas as pd
import matplotlib.pyplot as plt


# Fonctions pour le programme principal
def Moyenne(Serie):
    N=len(Serie)
    return sum(Serie)/N

def Variance(X):
    N=X.shape[0]
    Xm=sum(X)/N
    return sum((X-Xm)*(X-Xm))/N

def Covariance(Serie1,Serie2) : 
    return Moyenne((Serie1 - Moyenne(Serie1)) * (Serie2 - Moyenne(Serie2)))

def Correl(Serie1,Serie2) : 
    return Covariance(Serie1,Serie2) / (Variance(Serie1) * Variance(Serie2)) ** 0.5

def calculerMatriceParametres(arr_X,arr_Y) : 

    arr_TranspoX = arr_X.transpose()

    arr_resultat = lag.inv(arr_TranspoX @ arr_X)

    arr_resultat2 = arr_TranspoX @ arr_Y

    arr_resultatFinal = arr_resultat @ arr_resultat2

    return arr_resultatFinal

def CorrelMultiple(VarY,ErrMoy) :
    return (1-ErrMoy/VarY) ** 0.5

def ErreurQuadratiqueMoy(arr_matriceParamatres,li_SerieEndo,li_Serie1,li_Serie2,li_Serie3 ) :
    res = 0
    sommeYPred = 0
    for i in range(len(li_SerieEndo)) :
        YPred = li_Serie1[i] * arr_matriceParamatres[1,0] + li_Serie2[i] * arr_matriceParamatres[2,0] + li_Serie3[i] * arr_matriceParamatres[3,0] + arr_matriceParamatres[0,0]
        sommeYPred += (YPred - li_SerieEndo[i]) ** 2
    res = 1/len(li_SerieEndo) * sommeYPred
    return res
    
def creerMatriceVariableEndo(li_Serie) :
    return np.resize(np.array(li_Serie), (len(li_Serie),1))

def creerMatriceVariableExplicative(li_Serie1,li_Serie2,li_Serie3) : 
    N = len(li_Serie1)
    arr_matrice = np.ones((N,1))
    arr_s1 = np.array(li_Serie1).reshape(N ,1)
    arr_s2 = np.array(li_Serie2).reshape(N ,1)
    arr_s3 = np.array(li_Serie3).reshape(N ,1)
    arr_matriceFinal = np.concatenate((arr_matrice, arr_s1, arr_s2, arr_s3), axis=1)
    return arr_matriceFinal

# Partie principale du sujet

df_Donnees = pd.read_csv("/home/etuinfo/gfichantbell/Bureau/SAE204/Partie3/donneesvueparcoursup.csv")

df_num_donnees = df_Donnees._get_numeric_data()

arr_donnees = np.array(df_num_donnees)

li_pourcentage_mention_assez_bien = arr_donnees[:,0]

li_pourcentage_admise = arr_donnees[:,1]
variance_pourcentage_admise = Variance(li_pourcentage_admise)

li_taux_admission_formation = arr_donnees[:,2]

li_population_region= arr_donnees[:,3]

arr_matriceVariableEndo = creerMatriceVariableEndo(li_pourcentage_admise)

arr_matriceVariablesExpli = creerMatriceVariableExplicative(li_pourcentage_mention_assez_bien,li_taux_admission_formation,li_population_region)

# Partie Affichage des résultat
 
# Boîte à moustache 1
plt.boxplot(li_pourcentage_mention_assez_bien.tolist())
plt.title("Pourcentage de mention Assez-Bien par les admis")
plt.show()

# Boîte à moustache 2
plt.boxplot(li_pourcentage_admise .tolist())
plt.title("Pourcentage d'admise dans la formation")
plt.show()

# Boîte à moustache 3
plt.boxplot(li_taux_admission_formation.tolist())
plt.title("Taux d'admission à la formation")
plt.show()


# Nuage de points 1
coefCor1 = Correl(li_pourcentage_admise,li_pourcentage_mention_assez_bien)
print(coefCor1)
plt.plot(li_pourcentage_admise,li_pourcentage_mention_assez_bien,'.')
plt.ylabel("Pourcentage de mention Assez-Bien")
plt.xlabel("Pourcentage d'admise dans la formation")
plt.title("Pourcentage de mention Assez-Bien selon le pourcentage d'admise dans la formation")
plt.show()

# Nuage de point 2
coefCor2 = Correl(li_pourcentage_admise,li_taux_admission_formation)
print(coefCor2)
plt.plot(li_pourcentage_admise,li_taux_admission_formation,'.')
plt.ylabel("Taux d'admission de la formation")
plt.xlabel("Pourcentage d'admise dans la formation")
plt.title("Taux d'admission selon le pourcentage d'admise dans la formation")
plt.show()

# Nuage de point 3
coefCor3 = Correl(li_pourcentage_admise,li_population_region)
print(coefCor3)
plt.plot(li_pourcentage_admise,li_population_region,'.')
plt.ylabel("Population totale de la région où se situe la formation")
plt.xlabel("Pourcentage d'admise dans la formation")
plt.title("Population totale de la région où se situe la formation selon le pourcentage d'admise dans la formation")
plt.show()

#Partie Régression linéaire
matrice_parametres = calculerMatriceParametres(arr_matriceVariablesExpli,arr_matriceVariableEndo)
erreur_moyenne = ErreurQuadratiqueMoy(matrice_parametres, li_pourcentage_admise, li_pourcentage_mention_assez_bien, li_taux_admission_formation, li_population_region)
coefCorMultiple = CorrelMultiple(variance_pourcentage_admise,erreur_moyenne)
