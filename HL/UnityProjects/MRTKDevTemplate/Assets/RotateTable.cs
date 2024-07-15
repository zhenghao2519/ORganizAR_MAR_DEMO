using UnityEngine;

public class RotateTable : MonoBehaviour
{
    public Vector3 rotationAxis = Vector3.up;
    public float rotationSpeed = 50f;

    private float targetRotation = 35f;
    private float currentRotation = 0f;
    private bool isRotating = true;
    private int rotationDirection = 1;

    void Update()
    {
        if (isRotating)
        {
            float step = rotationSpeed * Time.deltaTime * rotationDirection;
            currentRotation += Mathf.Abs(step);
            transform.Rotate(rotationAxis, step);
            if (currentRotation >= targetRotation)
            {
                transform.Rotate(rotationAxis, (targetRotation - currentRotation) * rotationDirection);
                currentRotation = 0f;
                rotationDirection *= -1; // Switch direction after completing a 90-degree rotation
                if (rotationDirection > 0)
                {
                    rotationAxis = -rotationAxis; // Switch rotation axis for alternating rotations
                }
            }
        }
    }

    public void ToggleRotation(bool state)
    {
        isRotating = state;
    }
}
