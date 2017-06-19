/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <random>
#include <iostream>
#include <tuple>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

//************************************************
//*      Initialization
//************************************************

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    default_random_engine gen;
    
    
    //   ----------->
    //Step 1: Set the number of particles
    num_particles = 151;         //Vary This number as much as need
    // <--------------
    
    
    //Step 2: Initialize all the particles to first position
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    //Step 3: Add random Gaussian noise to each particle
    for (int i=0; i<num_particles;i++){
        Particle newPar;
        newPar.x = dist_x(gen);
        newPar.y = dist_y(gen);
        newPar.theta = dist_theta(gen);
        newPar.weight = 1.0;
        particles.push_back(newPar);
        weights.push_back(1);
        
    }
    //Final Step: Confirm I have initialized
    is_initialized = true;
    
}


//****************************************************
//*         Prediction
//****************************************************

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    //Step 1: Set my noises
	default_random_engine gen;
	normal_distribution<double> noise_x(0.0, std_pos[0]);
	normal_distribution<double> noise_y(0.0, std_pos[1]);
	normal_distribution<double> noise_theta(0.0, std_pos[2]);
    
    //Calculate the new particle position considering the noise
	for (int i = 0; i < num_particles; i++){
        if (fabs(yaw_rate) < 0.001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }
        else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }
        particles[i].x += noise_x(gen);
        particles[i].y += noise_y(gen);
        particles[i].theta += noise_theta(gen);
    }
}

//****************************************************
//*         Data Association
//****************************************************
 

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    
    double distX,distY,predX,predY,obsX,obsY,minDistSquare,minDist,distSquare,dist;
    std::vector<LandmarkObs> associations;
    
    //Step 1: Find the predicted measurement
    // Apply for all the observations (landmarks)
    for(int i=0;i<observations.size();i++){
        //obtain the closest object
        auto obs = observations[i];
        obsX = obs.x;
        obsY = obs.y;
        
        //get the first min distance to initialize
        auto minGlobal = predicted[0];
        
        predX = minGlobal.x;
        predY = minGlobal.y;
        distX = (predX-obsX);
        distY = (predY-obsY);
        minDistSquare =pow(distX,2)+pow(distY,2);
        minDist = pow(minDistSquare,0.5);
        
        for (int j=0;j<predicted.size();j++){
            auto predice =predicted[j];
            predX = predice.x;
            predY = predice.y;
            distX = (predX-obsX);
            distY = (predY-obsY);
            distSquare = pow(distX,2)+(pow(distY, 2));
            dist = pow(distSquare, 0.5);
            
            //apply squareMin because it's more accurate than regular min
            if (distSquare<minDistSquare){
                minGlobal = predicted[j];
            }
        }
        associations.push_back(minGlobal);
    }
}

//****************************************************
//*         Update Weights
//****************************************************

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

    
    double parX,parY,parTheta,landMarkX,landMarkY,landMarkID,distX,distY,obsX,obsY,transX,transY,dist,minGlobal;
    double xSigma, ySigma, xVariance, yVariance,xDelta,yDelta,associateX,associateY;
    int obsID;
    double weights_sum = 0;
    
    //variance: Var(x) = E[(X-u)^2]
    xSigma = std_landmark[0];
    ySigma = std_landmark[1];
    xVariance = pow(xSigma,2);
    yVariance = pow(ySigma,2);
    
    for (int i=0; i < num_particles; i++) {
        // predict measurements to all map landmarks
        Particle& particle = particles[i];
        long double weight = 1;
        
        //Particle Position
        parX = particles[i].x;
        parY = particles[i].y;
        parTheta = particles[i].theta;
        
        //Analize each observation
        vector<LandmarkObs> newPosition;
        std::vector<LandmarkObs> predLandMark;

        for (int j=0; j < observations.size(); j++) {
            
            //get the observations position and transform the position
            obsX = observations[j].x;
            obsY = observations[j].y;
            obsID = observations[j].id;
            
            //transform the position
            transX = parX + obsX*cos(parTheta)-obsY*sin(parTheta);
            transY = parY + obsX*sin(parTheta)+obsY*cos(parTheta);
            newPosition.push_back(LandmarkObs{obsID,transX,transY});
            
            minGlobal = sensor_range;
            dist = 0;
            
            // Data Associations
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
                LandmarkObs landMarkPred;
                landMarkX = map_landmarks.landmark_list[k].x_f;
                landMarkY = map_landmarks.landmark_list[k].y_f;
                landMarkID = map_landmarks.landmark_list[k].id_i;
                
                //calculate the distance between the landmark and the particle
                distX = fabs(landMarkX-transX);
                distY = fabs(landMarkY-transY);
                
                //check if the landmark is in the range
                    //Filter the landmarks so I only consider the one inside the sensor range
                    if (distX<=sensor_range){
                        if(distY<=sensor_range){
                            //cout << "LM: "<< landMarkID<< " / " << landMarkID << endl;
                            //Actualizamos los landmarks para la prediccion
                            predLandMark.push_back(landMarkPred);
                        }
                    }
                
                //dataAssociation(newPosition, predLandMark);
                // I didn't know how to make this part work, so I found the minGlobal again
                
                dist = pow(pow(distX,2)+pow(distY,2),.5);
                Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
            
                if (dist < minGlobal) {
                    
                    minGlobal = dist;
                    associateX = landmark.x_f;
                    associateY = landmark.y_f;
                }
            }
            
            xDelta = transX - associateX;
            yDelta = transY - associateY;
            weight *= (exp(-0.5*((pow(xDelta,2))/xVariance + pow(yDelta,2)/yVariance)))/(2*M_PI*xSigma*ySigma);
        }
        weights[i] = weight;
        weights_sum += weight;
    }
}

//****************************************************
//*         Resamble
//****************************************************


void ParticleFilter::resample() {
    default_random_engine gen;
    discrete_distribution<int> d(weights.begin(), weights.end());
    vector<Particle> newPar;
    
    for(unsigned i = 0; i < num_particles; i++)
    {
        auto ind = d(gen);
        newPar.push_back(std::move(particles[ind]));
    }
    
    particles = move(newPar);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}



