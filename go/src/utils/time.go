package utils

import (
	"fmt"
	"time"
)

type Config struct {
	Name   string
	Silent bool
}

func Time(config Config) func(...*time.Duration) {
	start := time.Now()

	return func(results ...*time.Duration) {
		for _, result := range results {
			*result = time.Since(start)
		}

		if !config.Silent {
			if config.Name == "" {
				config.Name = "Function"
			}

			fmt.Println(
				config.Name + " took " + time.Since(start).Round(time.Millisecond).String(),
			)
		}
	}
}
