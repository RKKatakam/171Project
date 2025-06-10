import React, { useState } from "react";
import { FaStar, FaStarHalf } from "react-icons/fa";
import "../styles/Survey.css";

export default function StarRate({ onRate }) {
    const [rating, setRating] = useState(null);
    const [hover, setHover] = useState(null);
    
    const handleClick = (value) => {
        setRating(value);
        onRate(value);
    };

    const handleMouseEnter = (value) => {
        setHover(value);
    };

    const handleMouseLeave = () => {
        setHover(null);
    };

    const renderStars = () => {
        const stars = [];
        const currentRating = hover || rating || 0;
        
        for (let i = 1; i <= 5; i++) {
            const isFull = currentRating >= i;
            const isHalf = currentRating >= i - 0.5 && currentRating < i;
            
            stars.push(
                <div key={i} className="star-container" style={{ position: "relative", cursor: "pointer" }}>
                    {/* Half star click area (left side) */}
                    <div
                        style={{
                            position: "absolute",
                            left: 0,
                            top: 0,
                            width: "50%",
                            height: "100%",
                            zIndex: 10
                        }}
                        onClick={() => handleClick(i - 0.5)}
                        onMouseEnter={() => handleMouseEnter(i - 0.5)}
                        onMouseLeave={handleMouseLeave}
                    />
                    
                    {/* Full star click area (right side) */}
                    <div
                        style={{
                            position: "absolute",
                            right: 0,
                            top: 0,
                            width: "50%",
                            height: "100%",
                            zIndex: 10
                        }}
                        onClick={() => handleClick(i)}
                        onMouseEnter={() => handleMouseEnter(i)}
                        onMouseLeave={handleMouseLeave}
                    />
                    
                    {/* Visual star */}
                    {isFull ? (
                        <FaStar size={35} color="#ffc107" />
                    ) : isHalf ? (
                        <FaStarHalf size={35} color="#ffc107" />
                    ) : (
                        <FaStar size={35} color="#e4e5e9" />
                    )}
                </div>
            );
        }
        
        return stars;
    };

    return (
        <div className="star-rating-container">
            <div className="stars-wrapper">
                {renderStars()}
            </div>
            <div className="rating-display">
                {rating ? `${rating}/5` : "Click to rate"}
            </div>
        </div>
    );
}

