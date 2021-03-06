﻿using System;
using UnityEngine;
using UnityStandardAssets.CrossPlatformInput;

namespace UnityStandardAssets.Vehicles.Car
{
    [RequireComponent(typeof(CarController))]
    public class CarRemoteControl : MonoBehaviour
    {
        private CarController m_Car; // the car controller we want to use

        public float SteeringAngle { get; set; }
        public float Acceleration { get; set; }
        private Steering s;

        void Awake()
        {
            // get the car controller
            Debug.Log("Car controller is awaken !!!!!");
            m_Car = GetComponent<CarController>();
        }

         void FixedUpdate()
        {
            m_Car.Move(SteeringAngle, Acceleration, Acceleration, 0f);
        }
    }
}

