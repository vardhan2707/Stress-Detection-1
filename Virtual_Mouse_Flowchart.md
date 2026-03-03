# Gesture Based Virtual Mouse System — Flowchart (from Batch - 7 rw- 04.pptx)

## Project Info
- **Title:** A Gesture Based Virtual Mouse System
- **Batch:** 17
- **Guide:** Mr. T. Krishna Murthy

---

## 1. Complete Implementation — System Pipeline

```mermaid
flowchart TD
    A[Webcam] --> B[Frame Processing]
    B --> C[Hand Detection / Landmark Model]
    C --> D[Cursor Mapping / Smoothing Filter]
    D --> E[System Control]
    E --> F[Computer UI]
```

## 2. Core Processing Flow

```mermaid
flowchart TD
    A[Start] --> B[Capture Frame via Webcam]
    B --> C[Process Frame]
    C --> D{Hand Detected?}
    D -->|Yes| E[Extract Hand Landmarks]
    D -->|No| B
    E --> F[Map Landmarks to Cursor Position]
    F --> G[Apply Smoothing Filter]
    G --> H{Recognize Gesture}
    H --> I[Execute Mouse Action]
    I --> B
    H --> B
```

## 3. Gesture Recognition & Control Flow

```mermaid
flowchart TD
    A[Hand Landmarks] --> B{Gesture Type?}
    B -->|Index Finger Extended| C[Cursor Movement]
    B -->|Pinch/Click| D[Click Operation]
    B -->|Scroll Gesture| E[Scroll Operation]
    B -->|Drag Gesture| F[Drag Operation]
    C --> G[Update Cursor Position]
    D --> H[Click Event]
    E --> I[Scroll Event]
    F --> J[Drag Event]
    G --> K[System Control]
    H --> K
    I --> K
    J --> K
    K --> L[Computer UI Update]
```

## 4. Evaluation Metrics Flow

```mermaid
flowchart TD
    A[System Evaluation] --> B[Hand Detection Accuracy]
    A --> C[Gesture Recognition Accuracy]
    A --> D[System Response Time]
    A --> E[Cursor Stability]
    A --> F[Error Rate]
    B --> G[Quantitative Results]
    C --> G
    D --> G
    E --> G
    F --> G
    G --> H{Meets Objectives?}
    H -->|Yes| I[Validated for Real-World Usage]
    H -->|No| J[Iterate & Improve]
```

## 5. Comparison: Traditional vs Proposed System

```mermaid
flowchart LR
    subgraph Traditional["Traditional Input Devices"]
        T1[Touch-based Control]
        T2[Limited Flexibility]
        T3[Fixed Deployment]
    end

    subgraph Proposed["Proposed Virtual Mouse"]
        P1[Touch-Free Gesture]
        P2[High Flexibility]
        P3[Easy Deployment]
    end

    Traditional -->|vs| Proposed
```

## 6. Overall System Architecture

```mermaid
flowchart TB
    subgraph Input["Input Layer"]
        Webcam[Webcam]
    end

    subgraph Processing["Processing Layer"]
        Frame[Frame Processing]
        Hand[Hand Detection / Landmark Model]
        Map[Cursor Mapping]
        Smooth[Smoothing Filter]
    end

    subgraph Output["Output Layer"]
        Control[System Control]
        UI[Computer UI]
    end

    Webcam --> Frame
    Frame --> Hand
    Hand --> Map
    Map --> Smooth
    Smooth --> Control
    Control --> UI
```

## 7. Key Achievements Flow

```mermaid
flowchart TD
    A[Project Objectives] --> B[Real-time Virtual Mouse]
    A --> C[Accurate Hand Detection]
    A --> D[Smooth Cursor Control]
    A --> E[Touch-Free Interaction]
    A --> F[High Accuracy & Low Response Time]

    B --> G[Achievement of Objectives]
    C --> G
    D --> G
    E --> G
    F --> G

    G --> H[Key Contributions]
    H --> I[Low-cost HCI System]
    H --> J[CV & ML for Hand Tracking]
    H --> K[Distance-based Gesture Recognition]
    H --> L[Accessible & Hygienic Alternative]
```

## 8. Implementation to Deployment Flow

```mermaid
flowchart LR
    A[Design] --> B[Develop]
    B --> C[Hand Detection]
    C --> D[Gesture Mapping]
    D --> E[Smoothing Algorithm]
    E --> F[Integration]
    F --> G[Testing]
    G --> H[Deployment]
    H --> I[Standard Webcam]
```

---

## Quick Reference: System Components

| Component | Purpose |
|-----------|---------|
| Webcam | Capture real-time video input |
| Frame Processing | Preprocess video frames |
| Hand Detection | Detect and track hand in frame |
| Landmark Model | Extract 21 hand landmark points |
| Cursor Mapping | Map landmarks to screen coordinates |
| Smoothing Filter | Stabilize cursor movement |
| System Control | Execute mouse events (click, scroll, drag) |
| Computer UI | Display and interact with applications |
