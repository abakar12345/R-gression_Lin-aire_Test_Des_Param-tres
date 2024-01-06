#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:15:49 2023

@author: abakar
"""
#########################################################################################################################
#                                          << IMPORTATION DES BIBLIOTHEQUES >>                                          #
#########################################################################################################################
from math import sqrt
from matplotlib import pyplot as plt
import scipy.stats as stats


#                    << IMPLEMENTATON DES FONCTIONS PRDEFINIES >>

def Range(debut, fin=None, etape=1): # foncton range()
    if fin is None:
        fin = debut
        debut =0                     # on initialise la première valeur à 0
    resultat=[]                      # on crée une liste vide pour stoker les valeurs par la suite
    i=debut                          # on initialise l'élément à stocker dans la liste à O
    while i<fin:
        resultat.append(i)           # ajout un element dans la liste 
        i += etape                   # on incrémente à chaque fois l'élément de 1 car etape=1
    return resultat


def Len(Objet):                     # fonction len()
    compteur = 0 
    for element in Objet:
        compteur += 1 
    return compteur 


def arrondir(nombre, decimales=0):          # fonction around()
    puissance = 10 ** decimales             # pour pouvoir extraire la decimale
    entier = int(nombre * puissance)        # recuperer la partie entiere
    decimal = nombre * puissance - entier   # extraction de la decimale à comparer avec 0.5
    if decimal >= 0.5:
        entier += 1         
    return entier / puissance           # on divise pour replacer la virgule a sa place

#########################################################################################################################
#                                                       PROGRAMME PRINCIPAl                                             #
#########################################################################################################################

#                                              << AVEC LES DEUX STAGIAIRES >>
x1=[3,4,6,7,9,10,9,11,12,13,15,4]
y1=[8,9,10,13,15,14,13,16,13,19,6,19]

droite_x=[0,10,15,20]
droite_y=[11.99,13.1,13.64,14.19]
plt.plot(droite_x,droite_y)

plt.scatter(x1,y1)

plt.title("Nuages des points avec les deux stagiares")
plt.xlabel("Valeurs de X")
plt.ylabel("Valeurs de Y")

plt.show()

#                                              << SANS LES DEUX STAGIAIRES >>
x=[3,4,6,7,9,10,9,11,12,13]
y=[8,9,10,13,15,14,13,16,13,19]

droite_x=[0,10,15,20]
droite_y=[5.47,14.47,18.97,23.47]
plt.plot(droite_x,droite_y)


plt.scatter(x,y)

plt.title("Nuages des points sans les deux stagiaires")
plt.xlabel("Valeurs de X")
plt.ylabel("Valeurs de y")

plt.show()

#########################################################################################################################
#                                                  <<  LES FONCTIONS  >>                                                #                        #
#########################################################################################################################

def moyenne(tab):
        somme =0
        for i in Range(Len(tab)):
            somme+=tab[i]
        somme= arrondir(somme/Len(tab),2)
        return somme 
print(f" AVEC TOUS LES STAGIAIRES (Moyenne) :\n >>> Moy(X) = {moyenne(x1)}; Moy(Y) = {moyenne(y1)}")
print(f" SANS LES DEUX STAGIAIRES (Moyenne) :\n >>> Moy(X) = {moyenne(x)}; Moy(Y) = {moyenne(y)}\n")

def variance(tab):
    var=0
    for i in Range(Len(tab)):
        var+=(tab[i]-moyenne(tab))**2
    var= arrondir(var/Len(tab),2)
    return var
print(f" AVEC TOUS LES STAGIAIRES (variance) :\n >>> Var(X) = {variance(x1)}; Var(Y) = {variance(y1)}")
print(f" SANS LES DEUX STAGIAIRES (variance) :\n >>> Var(X) = {variance(x)}; Var(Y) = {variance(y)}\n")


def ecart_type(tab):
    return arrondir(sqrt(variance(tab)), 2)
print(f" AVEC TOUS LES STAGIAIRES (Ecart-type) :\n >>> Ecart-type(X) = {ecart_type(x1)}; Ecart-type(Y) = {ecart_type(y1)}")
print(f" SANS LES DEUX STAGIAIRES (Ecart-type) :\n >>> Ecart-type(X) = {ecart_type(x)}; Ecart-type(Y) = {ecart_type(y)}\n")


def covariance(tab1,tab2):
    n=Len(tab1)
    somme=0
    for i in Range(n):
        somme+=(tab1[i]-moyenne(tab1))*(tab2[i]-moyenne(tab2))
    somme= arrondir(somme/n,2)
    return somme   
print(f" AVEC TOUS LES STAGIAIRES (covariance) :\n >>> Cov(X,Y) = {covariance(x1,y1)}")
print(f" SANS LES DEUX STAGIAIRES (covariance) :\n >>> Cov(X,Y) = {covariance(x,y)} \n")
         

def coeffient_correlation(tab1,tab2):
    return arrondir(covariance(tab1,tab2)/(ecart_type(tab1)*ecart_type(tab2)),2)
print(f" AVEC TOUS LES STAGIAIRES (coefficient de corrélation) :\n >>> r = {coeffient_correlation(x1,y1)}")
print(f" SANS LES DEUX STAGIAIRES (coefficient de corrélation) :\n >>> r = {coeffient_correlation(x,y)} \n")
         

def calcul_a1(tab1,tab2):
    return arrondir(covariance(tab1,tab2)/variance(tab1),2)
print(f" AVEC TOUS LES STAGIAIRES ( Calcul de a1) :\n >>> a1 = {calcul_a1(x1,y1)}")
print(f" SANS LES DEUX STAGIAIRES (Calcul de a1) :\n >>> a1 = {calcul_a1(x,y)} \n")
  

def calcul_a0(tab1,tab2):
    return arrondir(moyenne(tab2)-(calcul_a1(tab1,tab2)*moyenne(tab1)),2)
print(f" AVEC TOUS LES STAGIAIRES (Calcul de a0) :\n >>> a0 = {calcul_a0(x1,y1)}")
print(f" SANS LES DEUX STAGIAIRES (Calcul de a0) :\n >>> a0 = {calcul_a0(x,y)} \n")
  

def model_prediction(tab1,tab2,x):
    return arrondir(calcul_a0(tab1,tab2)+calcul_a1(tab1,tab2)*x,2)
#print(model_prediction(x,y,0))

def scr(tab1,tab2):
    n=Len(tab1)
    somme=0
    for i in Range(n):
        somme+= (tab2[i]-(calcul_a0(tab1,tab2)+(calcul_a1(tab1,tab2)*tab1[i])))**2
        somme = arrondir(somme, 2)
    return somme
print(f" AVEC TOUS LES STAGIAIRES (scr) :\n >>> SCR = {scr(x1,y1)}")
print(f" SANS LES DEUX STAGIAIRES (scr) :\n >>> SCR = {scr(x,y)} \n")
  

def sct(tab1,tab2):
    n=Len(tab1)
    somme=0
    for i in Range(n):
        somme+= arrondir((tab2[i]-moyenne(tab2))**2,2)
    return somme   
print(f" AVEC TOUS LES STAGIAIRES (sct) :\n >>> SCT = {sct(x1,y1)}")
print(f" SANS LES DEUX STAGIAIRES (sct) :\n >>> SCT = {sct(x,y)} \n")
   

def sce(tab1,tab2):
    diff = sct(tab1,tab2)-scr(tab1,tab2)
    resultat = arrondir(diff,2)
    return resultat 
print(f" AVEC TOUS LES STAGIAIRES (sce) :\n >>> SCE = {sce(x1,y1)}")
print(f" SANS LES DEUX STAGIAIRES (sce) :\n >>> SCE = {sce(x,y)} \n")

def R_au_carree(tab1,tab2):
    return arrondir(sce(tab1,tab2)/sct(tab1,tab2),2)
print(f" AVEC TOUS LES STAGIAIRES (coefficient de détermination) :\n >>> R² = {R_au_carree(x1,y1)}")
print(f" SANS LES DEUX STAGIAIRES (coefficient de détermination) :\n >>> R² = {R_au_carree(x,y)} \n")
 
######################################################################################################################### 
#                                                      << TEST DES LOIS>>                                               #    
#########################################################################################################################
"""
def khi(tab1,tab2):
    khi= arrondir(scr(tab1,tab2)/(scr(tab1,tab2)/(Len(tab1)-2)),2)
    return khi
print(f"VALEUR DE khi-2 (Tous les stagiaires) = {khi(x1,y1)}")
print(f"VALEUR DE khi-2 (Sans les deux stagiaires) = {khi(x,y)}\n")
"""

def khi(tab1, tab2):
    somme_Xk = 0
    somme_Yk = 0
    Tableau = []
    for i in Range(0, Len(tab1)):
        t = tab1[i] + tab2[i]
        somme_Xk += tab1[i]
        somme_Yk += tab2[i]
        Tableau.append(t)
    
    somme_K = somme_Xk + somme_Yk
    k0 = []
    k1 = []
    
    for i in Range(0, Len(Tableau)):
        e1 = (somme_Xk*Tableau[i])/somme_K
        e1 = arrondir(e1, 2)
        e2 = (somme_Yk*Tableau[i])/somme_K
        e2 = arrondir(e2, 2)
        k0.append(e1)
        k1.append(e2)
    
    valeur = 0
    for i in Range(0, Len(k0)):
        valeur += (((tab1[i]-k0[i])**2/k0[i]) + ((tab2[i]-k1[i])**2)/k1[i])
        valeur = arrondir(valeur, 2)
    return valeur
print(f"VALEUR DE khi-2 (Tous les stagiaires) = {khi(x1,y1)}")
print(f"VALEUR DE khi-2 (Sans les deux stagiaires) = {khi(x,y)}\n")


def chi2_theorique(tab1):
        n=Len(tab1)-2
        chi2_table= arrondir(stats.chi2.ppf(q=0.95,df=n),2)
        return chi2_table
       
def Hypothese_khi2(tab1,tab2):
    if khi(tab1,tab2) > chi2_theorique(tab1):
        print("L'Hypotese H0 est rejete au risque de 5% ,donc les deux variables sont liees\n")
    else :
        print("On ne peut pas rejete L'hypothese H0,Donc les deux Variables sont independants ou elles ont une liaison faible\n")  
        
print(f"\nTEST DE Khi-2 (Avec les deux Stagiaires) :\n=> ")
Hypothese_khi2(x1,y1)
print(f"TEST DE khi-2 (Sans les deux Stagiaires) :\n=> ")
Hypothese_khi2(x,y)
print("\n")

def fisher(tab1,tab2):
    F= arrondir((Len(tab2)-2)*(R_au_carree(tab1,tab2)/(1-R_au_carree(tab1,tab2))),2)
    return F
print(f"VALEUR DE FISHER (Tous les stagiaires) = {fisher(x1,y1)}")
print(f"VALEUR DE FISHER (Sans les deux stagiaires) = {fisher(x,y)}\n")

from scipy.stats import f
def Hypothese_Fisher(tab1,tab2):
    dfn=1
    dfd=Len(tab1)-2
    F_critique=f.ppf(0.95,dfn,dfd)
    if F_critique<fisher(tab1,tab2):
        print("L'Hypotese H0 est rejete au risque de 5% ,donc les deux variables sont liees\n")
    else :
        print("On ne peut pas rejete L'hypothese H0,Donc les deux Variables sont independants ou elles ont une liaison faible\n")
        

print("TEST DE FISHER (Tous les Stagiaires) : \n=> ")
Hypothese_Fisher(x1,y1)
print("TEST DE FISHER (Sans les deux Stagiaires) : \n=> ")
Hypothese_Fisher(x,y)
print("\n")


def studentb0(tab1,tab2):
    s1=scr(tab1,tab2)/(Len(tab1)-2)
    x=0
    for i in Range(Len(tab1)):
        x+=tab1[i]
    var=Len(tab1)*(Len(tab1)*variance(tab1))
    S=x/var
    S1=sqrt(s1*S)
    T= arrondir(calcul_a0(tab1,tab2)/S1,2)
    return T
print(f"VALEUR DE STUDENTb0 (Tous les stagiaires) = {studentb0(x1,y1)}")
print(f"VALEUR DE STUDENTb0 (Sans les deux stagiaires) = {studentb0(x,y)}\n")

def studentb1(tab1,tab2):
    s1=scr(tab1,tab2)/(Len(tab1)-2)
    var=(Len(tab1)*variance(tab1))
    S1=sqrt(s1/var)
    T= arrondir(calcul_a1(tab1,tab2)/S1,2)
    return T
print(f"VALEUR DE STUDENTb1 (Tous les stagiaires) = {studentb1(x1,y1)}")
print(f"VALEUR DE STUDENTb1 (Sans les deux stagiaires) = {studentb1(x,y)}\n")

from scipy.stats import t
def Hypothese_student(tab1,tab2):
    dl=Len(tab1)-2
    alpha=0.05
    t_critique=t.ppf(1-alpha/2,dl)
    if t_critique<studentb0(tab1,tab2):
        print("L'hyphotese H0 est rejete,le coefficient b0 n'est pas null")
    else :
          print("L'hyphotese H0 ne peut pas etre rejete,le coefficient b0 est pas null\n")
    if t_critique<studentb1(tab1,tab2):
        print("L'hyphotese H0 est rejete,le coefficient b1 n'est pas null")
    else :
          print("L'hyphotese H0 ne peut pas etre rejete,le coefficient b1 est  null\n")
          
print("TEST DE STUDENTS (Avec les deux Stagiaires) : \n=> ")
Hypothese_student(x1,y1)
print("TEST DE STUDENTS (Sans les deux Stagiaires) : \n=> ")
Hypothese_student(x,y)
print("\n")

#########################################################################################################################
#                                                 << TRACER DES COURBES >>                                              #
#########################################################################################################################

from scipy.stats import chi2
import numpy as np
def Courbe_khi2(tab1,tab2):
    df=Len(tab1)-2
    x=np.linspace(0,30,1000)
    y=chi2.pdf(x,df)
    khi2=khi(tab1,tab2)
    
    plt.plot(x,y)
    plt.axvline(chi2_theorique(tab1),color='r',linestyle='--' ,label='valeur critique')
    plt.axvline(khi2,color='g',linestyle='--', label='valeur reelle')
    plt.legend()
    plt.title("Courbe de Test de Khi-2")
    plt.xlabel("Valeurs de X2")
    plt.ylabel("Densite de Probabilite")
    plt.show()
    
Courbe_khi2(x1,y1)
Courbe_khi2(x,y)

from scipy.stats import t
def Courbe_Student0(tab1,tab2):
    df=Len(tab1)-2
    x=np.linspace(-5,20,1000)
    y=t.pdf(x,df)
    alpha=0.05
    t_critique=t.ppf(1-alpha/2,df)
    
    plt.plot(x,y)
    plt.axvline(x=studentb0(tab1,tab2),color='g',linestyle='--' ,label='valeur reelle')
    plt.axvline(x=t_critique,color='r',linestyle='--' ,label='valeur critique')
    plt.legend()
    plt.title("Courbe du test de Student")
    plt.xlabel("Valeur de t")
    plt.ylabel("Densite de probabilite")
    plt.show()
    

from scipy.stats import t
def Courbe_Student1(tab1,tab2):
    df=Len(tab1)-2
    x=np.linspace(-5,20,1000)
    y=t.pdf(x,df)
    alpha=0.05
    t_critique=t.ppf(1-alpha/2,df)
    
    plt.plot(x,y)
    plt.axvline(x=studentb1(tab1,tab2),color='g',linestyle='--' ,label='valeur reelle')
    plt.axvline(x=t_critique,color='r',linestyle='--' ,label='valeur critique')
    plt.legend()
    plt.title("Courbe du test de Student")
    plt.xlabel("Valeur de t")
    plt.ylabel("Densite de probabilite")
    plt.show()
    
#Avec les valeurs aberantes
Courbe_Student0(x1,y1)
Courbe_Student1(x1,y1)
#sans les valeurs aberantes
Courbe_Student0(x,y)
Courbe_Student1(x,y)

from scipy.stats import f
import numpy as np
def Courbe_Fisher(tab1,tab2):
    dfn=1
    dfd=Len(tab1)-2
    F_critique=f.ppf(0.95,dfn,dfd)
    x=np.linspace(0,50,1000)
    y=f.pdf(x,dfn,dfd)
    
    plt.plot(x,y)
    #plt.axvline(x=F_critique,color='g',linestyle='--')
    plt.axvline(x=fisher(tab1,tab2),color='g',linestyle='--' ,label='valeur reelle')
    plt.axvline(x=F_critique,color='r',linestyle='--' ,label='valeur critique')
    plt.legend()
    plt.title("Courbe du test de Fisher")
    plt.xlabel("Valeur de F")
    plt.ylabel("Densite de probabilite")
    plt.show()

#avec Les deux stagiaires
Courbe_Fisher(x1,y1)
#sans Les deux stagiaires
Courbe_Fisher(x,y)

#                                                          END CODING
